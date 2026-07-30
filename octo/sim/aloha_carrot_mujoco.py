"""Optional MuJoCo renderer for the dataset-aligned left-arm carrot task."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import DatasetAlignedController
from octo.sim.aloha_carrot_left import SceneState
from octo.sim.aloha_carrot_left import gripper_open_fraction
from octo.sim.aloha_carrot_left import rollout_metrics
from octo.sim.aloha_carrot_left import save_gif
from octo.sim.aloha_carrot_left import save_image


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ACT_ASSET_ROOT = _REPO_ROOT.parent / "act" / "assets"

_LEFT_JOINT_NAMES = (
    "vx300s_left/waist",
    "vx300s_left/shoulder",
    "vx300s_left/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_left/wrist_angle",
    "vx300s_left/wrist_rotate",
)
_LEFT_FINGER_JOINT = "vx300s_left/left_finger"
_RIGHT_FINGER_JOINT = "vx300s_left/right_finger"


def follower_gripper_to_act_finger(follower_value: float) -> Tuple[float, float]:
    """Map follower gripper units to ACT slide-joint finger positions."""

    fraction = gripper_open_fraction(follower_value)
    left = 0.021 + fraction * (0.057 - 0.021)
    return float(left), float(-left)


@dataclasses.dataclass(frozen=True)
class MujocoSmokeResult:
    xml_path: str
    qpos_shape: Tuple[int, ...]
    ctrl_shape: Tuple[int, ...]
    primary_shape: Tuple[int, ...]
    wrist_shape: Tuple[int, ...]
    body_names: Tuple[str, ...]
    no_right_arm: bool
    static_object_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xml_path": self.xml_path,
            "qpos_shape": list(self.qpos_shape),
            "ctrl_shape": list(self.ctrl_shape),
            "primary_shape": list(self.primary_shape),
            "wrist_shape": list(self.wrist_shape),
            "body_names": list(self.body_names),
            "no_right_arm": self.no_right_arm,
            "static_object_delta": self.static_object_delta,
        }


@dataclasses.dataclass(frozen=True)
class MujocoKinematicRolloutResult:
    xml_path: str
    success: bool
    max_reward: float
    steps: int
    final_state: str
    primary_frames: int
    wrist_frames: int
    no_right_arm: bool
    max_carrot_step_translation: float
    max_carrot_quat_delta: float
    pre_grasp_carrot_translation_std: float
    pre_grasp_carrot_quat_std: float
    post_place_carrot_translation_std: float
    post_place_carrot_quat_std: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xml_path": self.xml_path,
            "success": self.success,
            "max_reward": self.max_reward,
            "steps": self.steps,
            "final_state": self.final_state,
            "primary_frames": self.primary_frames,
            "wrist_frames": self.wrist_frames,
            "no_right_arm": self.no_right_arm,
            "max_carrot_step_translation": self.max_carrot_step_translation,
            "max_carrot_quat_delta": self.max_carrot_quat_delta,
            "pre_grasp_carrot_translation_std": self.pre_grasp_carrot_translation_std,
            "pre_grasp_carrot_quat_std": self.pre_grasp_carrot_quat_std,
            "post_place_carrot_translation_std": self.post_place_carrot_translation_std,
            "post_place_carrot_quat_std": self.post_place_carrot_quat_std,
        }


class AlohaCarrotMujocoSmoke:
    """Builds and renders a MuJoCo scene using the ACT ALOHA left arm."""

    def __init__(
        self,
        config: Optional[AlohaCarrotLeftConfig] = None,
        *,
        act_asset_root: Optional[Path | str] = None,
    ):
        self.config = config or AlohaCarrotLeftConfig()
        self.act_asset_root = Path(act_asset_root or _DEFAULT_ACT_ASSET_ROOT)
        self.xml_path: Optional[Path] = None

    def build(self, output_dir: Path | str) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._stage_act_assets(output_dir)
        self.xml_path = output_dir / "aloha_carrot_left_mujoco.xml"
        self.xml_path.write_text(self._build_xml(), encoding="utf-8")
        return self.xml_path

    def make_physics(self, output_dir: Path | str):
        xml_path = self.build(output_dir)
        os.environ.setdefault("MUJOCO_GL", "egl")
        try:
            from dm_control import mujoco
        except Exception as exc:
            raise RuntimeError("dm_control is required for MuJoCo validation") from exc
        return mujoco.Physics.from_xml_path(str(xml_path))

    def run(self, output_dir: Path | str) -> MujocoSmokeResult:
        output_dir = Path(output_dir)
        physics = self.make_physics(output_dir)
        self.apply_reset(physics)

        primary = physics.render(height=480, width=640, camera_id="overhead_cam")
        wrist = physics.render(height=480, width=640, camera_id="wrist_cam_left")
        save_image(output_dir / "mujoco_primary_reset.png", primary)
        save_image(output_dir / "mujoco_wrist_reset.png", wrist)

        carrot_before = self._body_xpos(physics, "carrot_body").copy()
        for _ in range(8):
            physics.step()
        carrot_after = self._body_xpos(physics, "carrot_body").copy()

        body_names = tuple(self._body_names(physics))
        result = MujocoSmokeResult(
            xml_path=str(self.xml_path),
            qpos_shape=tuple(int(v) for v in physics.data.qpos.shape),
            ctrl_shape=tuple(int(v) for v in physics.data.ctrl.shape),
            primary_shape=tuple(int(v) for v in primary.shape),
            wrist_shape=tuple(int(v) for v in wrist.shape),
            body_names=body_names,
            no_right_arm=not any(_is_right_arm_body(name) for name in body_names),
            static_object_delta=float(np.linalg.norm(carrot_after - carrot_before)),
        )
        return result

    def run_scripted_rollout(
        self,
        output_dir: Path | str,
        *,
        max_steps: int = 160,
        render_stride: int = 2,
    ) -> MujocoKinematicRolloutResult:
        output_dir = Path(output_dir)
        physics = self.make_physics(output_dir)
        ik = _LeftArmIk(physics)
        sim = AlohaCarrotLeftSim(self.config)
        controller = DatasetAlignedController(self.config)
        states = sim.rollout_scripted(max_steps=max_steps, controller=controller)

        primary_frames: List[np.ndarray] = []
        wrist_frames: List[np.ndarray] = []
        for idx, state in enumerate(states):
            left_joint_qpos = ik.solve(state.ee_pos)
            self.apply_kinematic_state(
                physics,
                state,
                left_joint_qpos=left_joint_qpos,
            )
            if idx % render_stride == 0 or idx == len(states) - 1:
                primary_frames.append(
                    physics.render(height=480, width=640, camera_id="overhead_cam")
                )
                wrist_frames.append(
                    physics.render(height=480, width=640, camera_id="wrist_cam_left")
                )

        save_image(output_dir / "mujoco_primary_final.png", primary_frames[-1])
        save_image(output_dir / "mujoco_wrist_final.png", wrist_frames[-1])
        save_gif(output_dir / "mujoco_primary_rollout.gif", primary_frames)
        save_gif(output_dir / "mujoco_wrist_rollout.gif", wrist_frames)

        metrics = rollout_metrics(states)
        body_names = tuple(self._body_names(physics))
        return MujocoKinematicRolloutResult(
            xml_path=str(self.xml_path),
            success=bool(metrics["success"]),
            max_reward=float(metrics["max_reward"]),
            steps=int(metrics["steps"]),
            final_state=str(metrics["final_state"]),
            primary_frames=len(primary_frames),
            wrist_frames=len(wrist_frames),
            no_right_arm=not any(_is_right_arm_body(name) for name in body_names),
            max_carrot_step_translation=float(metrics["max_carrot_step_translation"]),
            max_carrot_quat_delta=float(metrics["max_carrot_quat_delta"]),
            pre_grasp_carrot_translation_std=float(
                metrics["pre_grasp_carrot_translation_std"]
            ),
            pre_grasp_carrot_quat_std=float(metrics["pre_grasp_carrot_quat_std"]),
            post_place_carrot_translation_std=float(
                metrics["post_place_carrot_translation_std"]
            ),
            post_place_carrot_quat_std=float(metrics["post_place_carrot_quat_std"]),
        )

    def apply_reset(self, physics) -> None:
        sim = AlohaCarrotLeftSim(self.config)
        self.apply_kinematic_state(physics, sim.state)

    def apply_kinematic_state(
        self,
        physics,
        state: SceneState,
        *,
        left_joint_qpos: Optional[Sequence[float]] = None,
    ) -> None:
        joint_qpos = state.left_qpos[:6] if left_joint_qpos is None else left_joint_qpos
        for joint_name, value in zip(_LEFT_JOINT_NAMES, joint_qpos):
            _set_joint_qpos(physics, joint_name, float(value))
        left_finger, right_finger = follower_gripper_to_act_finger(state.gripper)
        _set_joint_qpos(physics, _LEFT_FINGER_JOINT, left_finger)
        _set_joint_qpos(physics, _RIGHT_FINGER_JOINT, right_finger)

        carrot_qpos = np.concatenate([state.carrot_pos, state.carrot_quat])
        _set_free_joint_qpos(physics, "carrot_joint", carrot_qpos)

        ctrl = np.array(list(joint_qpos) + [left_finger, right_finger])
        physics.data.ctrl[: len(ctrl)] = ctrl
        physics.data.qvel[:] = 0.0
        physics.forward()

    def _stage_act_assets(self, output_dir: Path) -> None:
        if not self.act_asset_root.exists():
            raise FileNotFoundError(
                f"ACT asset root not found: {self.act_asset_root}"
            )
        for src in self.act_asset_root.iterdir():
            if src.is_file():
                shutil.copy2(src, output_dir / src.name)
        _patch_left_arm_xml(output_dir / "vx300s_left.xml", self.config)

    def _build_xml(self) -> str:
        cfg = self.config
        mat_x, mat_y, mat_z = cfg.mat_center
        mat_w, mat_h = cfg.mat_size
        cup_x, cup_y, cup_z = cfg.cup_pos
        plate_x, plate_y, plate_z = cfg.plate_pos
        carrot_x, carrot_y, carrot_z = cfg.carrot_pos
        carrot_quat = _space(cfg.carrot_quat)
        return f"""<mujoco model="aloha_carrot_left_dataset_aligned">
  <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5" autolimits="true"/>
  <option timestep="0.02" gravity="0 0 0" iterations="50" tolerance="1e-10" solver="Newton" cone="elliptic"/>
  <size njmax="1000" nconmax="300"/>

  <include file="vx300s_dependencies.xml"/>

  <asset>
    <texture name="green_sky" type="skybox" builtin="flat" rgb1="0.0 0.42 0.30" width="512" height="512"/>
    <material name="wood" rgba="0.76 0.66 0.50 1"/>
    <material name="green_backdrop" rgba="0.0 0.42 0.30 1"/>
    <material name="pink_mat" rgba="1.0 0.28 0.50 0.58"/>
    <material name="cup_blue" rgba="0.22 0.80 0.92 1"/>
    <material name="plate_green" rgba="0.30 0.60 0.40 1"/>
    <material name="carrot_orange" rgba="0.95 0.42 0.08 1"/>
    <material name="leaf_green" rgba="0.20 0.52 0.20 1"/>
    <material name="arm_black" rgba="0.005 0.005 0.005 1"/>
  </asset>

  <worldbody>
    <light castshadow="false" directional="true" diffuse="0.45 0.45 0.45" pos="-1 -1 1" dir="1 1 -1"/>
    <light castshadow="false" directional="true" diffuse="0.35 0.35 0.35" pos="1 -1 1" dir="-1 1 -1"/>
    <light castshadow="false" directional="true" diffuse="0.25 0.25 0.25" pos="0 1 1" dir="0 -1 -1"/>

    <body name="table" pos="0 0.18 0">
      <geom name="table_top" type="box" size="0.90 0.52 0.018" pos="0 0 -0.018" material="wood" contype="1" conaffinity="1"/>
    </body>
    <geom name="green_back_wall" type="box" size="1.00 0.018 0.52" pos="0 0.70 0.43" material="green_backdrop" contype="0" conaffinity="0"/>
    <geom name="green_left_wall" type="box" size="0.018 0.58 0.48" pos="-0.92 0.20 0.40" material="green_backdrop" contype="0" conaffinity="0"/>
    <geom name="mat" type="box" size="{mat_w / 2:.6f} {mat_h / 2:.6f} 0.002" pos="{mat_x:.6f} {mat_y:.6f} {mat_z:.6f}" material="pink_mat" contype="0" conaffinity="0"/>

    <body name="primary_focus" pos="0.0 0.168 0.04">
      <site name="primary_focus_site" size="0.01" rgba="1 0 0 0"/>
    </body>
    <camera name="overhead_cam" pos="0 -0.420000 0.620000" mode="targetbody" target="primary_focus" fovy="{cfg.primary_camera_fovy:.6f}"/>

    <include file="vx300s_left.xml"/>

    <body name="cup_body" pos="{cup_x:.6f} {cup_y:.6f} {cup_z:.6f}">
      <geom name="cup" type="cylinder" size="0.075 0.070" material="cup_blue" contype="1" conaffinity="1"/>
      <geom name="cup_handle" type="box" size="0.012 0.007 0.032" pos="-0.086 0 0.005" material="cup_blue" contype="0" conaffinity="0"/>
    </body>
    <body name="plate_body" pos="{plate_x:.6f} {plate_y:.6f} {plate_z:.6f}">
      <geom name="plate" type="cylinder" size="0.062 0.008" material="plate_green" contype="1" conaffinity="1"/>
    </body>
    <body name="carrot_body" pos="{carrot_x:.6f} {carrot_y:.6f} {carrot_z:.6f}" quat="{carrot_quat}">
      <inertial pos="0 0 0" mass="0.05" diaginertia="0.0001 0.0001 0.0001"/>
      <joint name="carrot_joint" type="free"/>
      <geom name="carrot" type="capsule" size="0.018 0.055" material="carrot_orange" contype="0" conaffinity="0"/>
      <geom name="carrot_leaf" type="ellipsoid" size="0.018 0.026 0.010" pos="0 0 -0.060" material="leaf_green" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <position name="left_waist" joint="vx300s_left/waist" kp="43" ctrlrange="-3.14158 3.14158"/>
    <position name="left_shoulder" joint="vx300s_left/shoulder" kp="265" ctrlrange="-1.85005 1.25664"/>
    <position name="left_elbow" joint="vx300s_left/elbow" kp="227" ctrlrange="-1.76278 1.6057"/>
    <position name="left_forearm_roll" joint="vx300s_left/forearm_roll" kp="78" ctrlrange="-3.14158 3.14158"/>
    <position name="left_wrist_angle" joint="vx300s_left/wrist_angle" kp="37" ctrlrange="-1.8675 2.23402"/>
    <position name="left_wrist_rotate" joint="vx300s_left/wrist_rotate" kp="10.4" ctrlrange="-3.14158 3.14158"/>
    <position name="left_left_finger" joint="vx300s_left/left_finger" kp="2000" ctrlrange="0.021 0.057"/>
    <position name="left_right_finger" joint="vx300s_left/right_finger" kp="2000" ctrlrange="-0.057 -0.021"/>
  </actuator>
</mujoco>
"""

    def _body_xpos(self, physics, body_name: str) -> np.ndarray:
        body_id = physics.model.name2id(body_name, "body")
        return np.asarray(physics.data.xpos[body_id], dtype=np.float64)

    def _body_names(self, physics) -> List[str]:
        names = []
        for body_id in range(physics.model.nbody):
            name = physics.model.id2name(body_id, "body")
            if name:
                names.append(name)
        return names


def write_report(
    path: Path | str,
    *,
    smoke: MujocoSmokeResult,
    rollout: MujocoKinematicRolloutResult,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"smoke": smoke.to_dict(), "rollout": rollout.to_dict()}, f, indent=2)


class _LeftArmIk:
    def __init__(self, physics):
        import mujoco

        self._physics = physics
        self._mujoco = mujoco
        self._site_id = physics.model.name2id("left_cam_focus", "site")
        self._joint_ids = [
            physics.model.name2id(joint_name, "joint")
            for joint_name in _LEFT_JOINT_NAMES
        ]
        self._qpos_addr = np.array(
            [physics.model.jnt_qposadr[joint_id] for joint_id in self._joint_ids],
            dtype=np.int64,
        )
        self._dof_addr = np.array(
            [physics.model.jnt_dofadr[joint_id] for joint_id in self._joint_ids],
            dtype=np.int64,
        )
        ranges = np.asarray(
            [physics.model.jnt_range[joint_id] for joint_id in self._joint_ids],
            dtype=np.float64,
        )
        self._lower = ranges[:, 0]
        self._upper = ranges[:, 1]
        self._qpos = physics.data.qpos[self._qpos_addr].copy()

    def solve(
        self,
        target: Sequence[float],
        *,
        max_iters: int = 36,
        tolerance: float = 0.018,
        damping: float = 0.030,
    ) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        qpos = self._qpos.copy()
        for _ in range(max_iters):
            self._physics.data.qpos[self._qpos_addr] = qpos
            self._physics.forward()
            site_pos = np.asarray(
                self._physics.data.site_xpos[self._site_id],
                dtype=np.float64,
            )
            error = target - site_pos
            if np.linalg.norm(error) <= tolerance:
                break

            jacp = np.zeros((3, self._physics.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self._physics.model.nv), dtype=np.float64)
            self._mujoco.mj_jacSite(
                self._physics.model.ptr,
                self._physics.data.ptr,
                jacp,
                jacr,
                self._site_id,
            )
            jac = jacp[:, self._dof_addr]
            lhs = jac @ jac.T + np.eye(3) * damping
            delta = jac.T @ np.linalg.solve(lhs, error)
            qpos = np.clip(qpos + np.clip(delta, -0.08, 0.08), self._lower, self._upper)

        self._qpos = qpos
        return qpos.copy()


def _set_joint_qpos(physics, joint_name: str, value: float) -> None:
    joint_id = physics.model.name2id(joint_name, "joint")
    address = int(physics.model.jnt_qposadr[joint_id])
    physics.data.qpos[address] = value


def _set_free_joint_qpos(physics, joint_name: str, values: Sequence[float]) -> None:
    joint_id = physics.model.name2id(joint_name, "joint")
    address = int(physics.model.jnt_qposadr[joint_id])
    physics.data.qpos[address : address + 7] = np.asarray(values, dtype=np.float64)


def _patch_left_arm_xml(path: Path, config: AlohaCarrotLeftConfig) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    for camera in root.iter("camera"):
        if camera.attrib.get("name") == "left_wrist":
            camera.set("name", "wrist_cam_left")
            camera.set("mode", "targetbody")
            camera.set("target", "carrot_body")
            camera.set("pos", "-0.1 0 0.16")
            camera.set("fovy", f"{config.wrist_camera_fovy:.6f}")
            for attr in ("xyaxes", "euler", "quat", "focal", "resolution", "sensorsize"):
                camera.attrib.pop(attr, None)
    for geom in root.iter("geom"):
        if geom.attrib.get("name", "").startswith("vx300s_left/"):
            geom.set("rgba", "0.005 0.005 0.005 1")
    tree.write(path, encoding="unicode")


def _space(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _is_right_arm_body(name: str) -> bool:
    return (
        name == "vx300s_right"
        or name.startswith("vx300s_right/")
        or name == "right"
        or name.startswith("right/")
    )
