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
from PIL import Image
from PIL import ImageDraw

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import DatasetAlignedController
from octo.sim.aloha_carrot_left import SceneState
from octo.sim.aloha_carrot_left import source_aligned_timeline
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
    initial_ee_pos: Tuple[float, ...]
    configured_initial_ee_pos: Tuple[float, ...]
    initial_ee_error: float

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
            "initial_ee_pos": list(self.initial_ee_pos),
            "configured_initial_ee_pos": list(self.configured_initial_ee_pos),
            "initial_ee_error": self.initial_ee_error,
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
    max_ee_tracking_error: float
    max_left_joint_step: float
    max_wrist_camera_rotation_step_deg: float
    final_reference_joint_error: float

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
            "max_ee_tracking_error": self.max_ee_tracking_error,
            "max_left_joint_step": self.max_left_joint_step,
            "max_wrist_camera_rotation_step_deg": (
                self.max_wrist_camera_rotation_step_deg
            ),
            "final_reference_joint_error": self.final_reference_joint_error,
        }


@dataclasses.dataclass(frozen=True)
class MujocoSourceTimelineResult:
    frame_names: Tuple[str, ...]
    labels: Tuple[str, ...]
    object_phases: Tuple[str, ...]
    primary_shapes: Tuple[Tuple[int, ...], ...]
    wrist_shapes: Tuple[Tuple[int, ...], ...]
    no_right_arm: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_names": list(self.frame_names),
            "labels": list(self.labels),
            "object_phases": list(self.object_phases),
            "primary_shapes": [list(shape) for shape in self.primary_shapes],
            "wrist_shapes": [list(shape) for shape in self.wrist_shapes],
            "no_right_arm": self.no_right_arm,
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
        initial_ee_pos = self._site_xpos(physics, "left_cam_focus")
        configured_initial_ee_pos = np.asarray(
            self.config.initial_ee_pos,
            dtype=np.float64,
        )
        result = MujocoSmokeResult(
            xml_path=str(self.xml_path),
            qpos_shape=tuple(int(v) for v in physics.data.qpos.shape),
            ctrl_shape=tuple(int(v) for v in physics.data.ctrl.shape),
            primary_shape=tuple(int(v) for v in primary.shape),
            wrist_shape=tuple(int(v) for v in wrist.shape),
            body_names=body_names,
            no_right_arm=not any(_is_right_arm_body(name) for name in body_names),
            static_object_delta=float(np.linalg.norm(carrot_after - carrot_before)),
            initial_ee_pos=tuple(float(value) for value in initial_ee_pos),
            configured_initial_ee_pos=tuple(
                float(value) for value in configured_initial_ee_pos
            ),
            initial_ee_error=float(
                np.linalg.norm(initial_ee_pos - configured_initial_ee_pos)
            ),
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
        self.apply_reset(physics)
        ik = _LeftArmIk(physics)
        sim = AlohaCarrotLeftSim(self.config)
        controller = DatasetAlignedController(self.config)
        states = sim.rollout_scripted(max_steps=max_steps, controller=controller)

        primary_frames: List[np.ndarray] = []
        wrist_frames: List[np.ndarray] = []
        left_joint_history: List[np.ndarray] = []
        wrist_rotation_history: List[np.ndarray] = []
        ee_tracking_errors: List[float] = []
        wrist_camera_id = physics.model.name2id("wrist_cam_left", "camera")
        for idx, state in enumerate(states):
            left_joint_qpos = (
                np.asarray(self.config.left_home_qpos[:6], dtype=np.float64)
                if idx == 0
                else ik.solve(
                    state.ee_pos,
                    posture_target=_source_posture_target(state, self.config),
                    posture_weight=0.12 if state.success else 0.08,
                    locked_qpos=_locked_joint_target(state, self.config),
                )
            )
            self.apply_kinematic_state(
                physics,
                state,
                left_joint_qpos=left_joint_qpos,
            )
            left_joint_history.append(np.asarray(left_joint_qpos).copy())
            wrist_rotation_history.append(
                np.asarray(physics.data.cam_xmat[wrist_camera_id]).reshape(3, 3).copy()
            )
            ee_tracking_errors.append(
                float(
                    np.linalg.norm(
                        self._site_xpos(physics, "left_cam_focus") - state.ee_pos
                    )
                )
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
        joint_history = np.stack(left_joint_history)
        max_left_joint_step = float(
            np.abs(np.diff(joint_history, axis=0)).max()
            if len(joint_history) > 1
            else 0.0
        )
        max_wrist_camera_rotation_step_deg = _max_rotation_step_degrees(
            wrist_rotation_history
        )
        final_reference_qpos = np.asarray(
            (
                self.config.source_qpos[-1][:6]
                if len(self.config.source_qpos) == 6
                else self.config.left_home_qpos[:6]
            ),
            dtype=np.float64,
        )
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
            max_ee_tracking_error=max(ee_tracking_errors, default=0.0),
            max_left_joint_step=max_left_joint_step,
            max_wrist_camera_rotation_step_deg=max_wrist_camera_rotation_step_deg,
            final_reference_joint_error=float(
                np.abs(joint_history[-1] - final_reference_qpos).max()
            ),
        )

    def render_source_timeline(
        self,
        output_dir: Path | str,
        *,
        max_steps: int = 160,
    ) -> MujocoSourceTimelineResult:
        output_dir = Path(output_dir)
        physics = self.make_physics(output_dir)
        self.apply_reset(physics)
        ik = _LeftArmIk(physics)
        sim = AlohaCarrotLeftSim(self.config)
        states = sim.rollout_scripted(max_steps=max_steps)
        timeline = source_aligned_timeline(states, self.config)

        timeline_by_index = {item.state_index: item for item in timeline}
        rendered_primary: Dict[str, Image.Image] = {}
        rendered_wrist: Dict[str, Image.Image] = {}
        rendered_primary_shapes: Dict[str, Tuple[int, ...]] = {}
        rendered_wrist_shapes: Dict[str, Tuple[int, ...]] = {}
        for idx, state in enumerate(states):
            item = timeline_by_index.get(idx)
            source_posture = None
            if (
                item is not None
                and int(item.frame_name) >= 5
                and len(self.config.source_qpos) == 6
            ):
                source_posture = np.asarray(
                    self.config.source_qpos[int(item.frame_name) - 1][:6],
                    dtype=np.float64,
                )
            left_joint_qpos = (
                np.asarray(self.config.left_home_qpos[:6], dtype=np.float64)
                if idx == 0
                else ik.solve(
                    state.ee_pos,
                    posture_target=(
                        source_posture
                        if source_posture is not None
                        else _source_posture_target(state, self.config)
                    ),
                    posture_weight=0.32 if source_posture is not None else 0.08,
                    locked_qpos=_locked_joint_target(state, self.config),
                )
            )
            self.apply_kinematic_state(
                physics,
                state,
                left_joint_qpos=left_joint_qpos,
            )
            if item is None:
                continue
            primary = physics.render(
                height=480,
                width=640,
                camera_id="overhead_cam",
            )
            wrist = physics.render(
                height=480,
                width=640,
                camera_id="wrist_cam_left",
            )
            save_image(output_dir / f"mujoco_high_{item.frame_name}.png", primary)
            save_image(output_dir / f"mujoco_wrist_{item.frame_name}.png", wrist)
            rendered_primary[item.frame_name] = Image.fromarray(primary)
            rendered_wrist[item.frame_name] = Image.fromarray(wrist)
            rendered_primary_shapes[item.frame_name] = tuple(
                int(value) for value in primary.shape
            )
            rendered_wrist_shapes[item.frame_name] = tuple(
                int(value) for value in wrist.shape
            )

        primary_images = [rendered_primary[item.frame_name] for item in timeline]
        wrist_images = [rendered_wrist[item.frame_name] for item in timeline]
        primary_shapes = [rendered_primary_shapes[item.frame_name] for item in timeline]
        wrist_shapes = [rendered_wrist_shapes[item.frame_name] for item in timeline]

        _write_source_timeline_contact_sheet(
            output_dir / "mujoco_source_timeline_contact_sheet.png",
            source_dir=Path(self.config.source_reference_dir),
            primary_images=primary_images,
            wrist_images=wrist_images,
        )
        body_names = tuple(self._body_names(physics))
        return MujocoSourceTimelineResult(
            frame_names=tuple(item.frame_name for item in timeline),
            labels=tuple(item.label for item in timeline),
            object_phases=tuple(item.state.object_phase for item in timeline),
            primary_shapes=tuple(primary_shapes),
            wrist_shapes=tuple(wrist_shapes),
            no_right_arm=not any(_is_right_arm_body(name) for name in body_names),
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
            raise FileNotFoundError(f"ACT asset root not found: {self.act_asset_root}")
        for src in self.act_asset_root.iterdir():
            if src.is_file():
                shutil.copy2(src, output_dir / src.name)
        _write_wood_texture(output_dir / "aloha_wood_texture.png", self.config.seed)
        _write_cloth_texture(output_dir / "aloha_cloth_texture.png")
        _patch_left_arm_xml(output_dir / "vx300s_left.xml", self.config)

    def _build_xml(self) -> str:
        cfg = self.config
        mat_x, mat_y, mat_z = cfg.mat_center
        mat_w, mat_h = cfg.mat_size
        mat_half_w, mat_half_h = mat_w / 2, mat_h / 2
        border_width = 0.010
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
    <texture name="green_sky" type="skybox" builtin="flat" rgb1="0.0 0.70 0.56" width="512" height="512"/>
    <texture name="wood_tex" type="2d" file="aloha_wood_texture.png"/>
    <texture name="cloth_tex" type="2d" file="aloha_cloth_texture.png"/>
    <material name="wood" texture="wood_tex" texrepeat="1 1" texuniform="true" reflectance="0.03" shininess="0.08"/>
    <material name="green_backdrop" texture="cloth_tex" texrepeat="1 1" rgba="0.0 1.0 0.80 1" reflectance="0.02"/>
    <material name="pink_mat" rgba="1.0 0.70 0.85 0.78" reflectance="0.02"/>
    <material name="pink_border" rgba="1.0 0.45 0.70 0.55" reflectance="0.02"/>
    <material name="cup_blue" rgba="0.25 0.82 1.0 1" emission="0.45" specular="0.45" shininess="0.55"/>
    <material name="cup_inner" rgba="0.06 0.40 0.57 1" specular="0.25" shininess="0.35"/>
    <material name="cup_logo" rgba="0.78 0.94 1.0 1" specular="0.20" shininess="0.30"/>
    <material name="cup_logo_dark" rgba="0.05 0.36 0.58 1" specular="0.20" shininess="0.30"/>
    <material name="plate_green" rgba="0.68 0.98 0.88 1" emission="0.08" specular="0.28" shininess="0.40"/>
    <material name="plate_rim" rgba="0.52 0.82 0.68 1" emission="0.08" specular="0.22" shininess="0.32"/>
    <material name="carrot_orange" rgba="1.0 0.62 0.44 1" emission="0.12" specular="0.38" shininess="0.48"/>
    <material name="carrot_seam" rgba="0.72 0.20 0.08 1" specular="0.32" shininess="0.40"/>
    <material name="leaf_green" rgba="0.20 0.48 0.20 1" specular="0.20" shininess="0.25"/>
    <material name="arm_black" rgba="0.018 0.022 0.025 1" specular="0.55" shininess="0.72"/>
  </asset>

  <visual>
    <headlight ambient="0.18 0.18 0.18" diffuse="0.28 0.28 0.28" specular="0.30 0.30 0.30"/>
    <quality shadowsize="4096"/>
  </visual>

  <worldbody>
    <light castshadow="true" directional="true" diffuse="0.45 0.45 0.45" specular="0.22 0.22 0.22" pos="-0.7 -0.8 1.6" dir="0.35 0.45 -1"/>
    <light castshadow="false" directional="true" diffuse="0.16 0.16 0.16" pos="0.8 -0.2 1.2" dir="-0.45 0.15 -1"/>

    <geom name="green_floor" type="box" size="1.20 1.20 0.020" pos="0 0.24 -0.058" material="green_backdrop" contype="0" conaffinity="0"/>
    <body name="table" pos="0 0.25 0">
      <geom name="table_top" type="box" size="0.57 0.60 0.018" pos="0 0 -0.018" material="wood" contype="1" conaffinity="1"/>
    </body>
    <geom name="green_back_wall" type="box" size="1.20 0.018 0.62" pos="0 0.87 0.54" material="green_backdrop" contype="0" conaffinity="0"/>
    <geom name="mat" type="box" size="{mat_half_w:.6f} {mat_half_h:.6f} 0.002" pos="{mat_x:.6f} {mat_y:.6f} {mat_z:.6f}" material="pink_mat" contype="0" conaffinity="0"/>
    <geom name="mat_border_left" type="box" size="{border_width:.6f} {mat_half_h:.6f} 0.0015" pos="{mat_x - mat_half_w + border_width:.6f} {mat_y:.6f} {mat_z + 0.003:.6f}" material="pink_border" contype="0" conaffinity="0"/>
    <geom name="mat_border_right" type="box" size="{border_width:.6f} {mat_half_h:.6f} 0.0015" pos="{mat_x + mat_half_w - border_width:.6f} {mat_y:.6f} {mat_z + 0.003:.6f}" material="pink_border" contype="0" conaffinity="0"/>
    <geom name="mat_border_back" type="box" size="{mat_half_w:.6f} {border_width:.6f} 0.0015" pos="{mat_x:.6f} {mat_y + mat_half_h - border_width:.6f} {mat_z + 0.003:.6f}" material="pink_border" contype="0" conaffinity="0"/>
    <geom name="mat_border_front" type="box" size="{mat_half_w:.6f} {border_width:.6f} 0.0015" pos="{mat_x:.6f} {mat_y - mat_half_h + border_width:.6f} {mat_z + 0.003:.6f}" material="pink_border" contype="0" conaffinity="0"/>

    <body name="primary_focus" pos="0.0 0.168 0.04">
      <site name="primary_focus_site" size="0.01" rgba="1 0 0 0"/>
    </body>
    <camera name="overhead_cam" pos="0 -0.420000 0.620000" mode="targetbody" target="primary_focus" fovy="{cfg.primary_camera_fovy:.6f}"/>

    <include file="vx300s_left.xml"/>

    <body name="cup_body" pos="{cup_x:.6f} {cup_y:.6f} {cup_z:.6f}">
      <geom name="cup" type="cylinder" size="0.050 0.058" material="cup_blue" contype="1" conaffinity="1"/>
      <geom name="cup_inner" type="cylinder" size="0.040 0.0015" pos="0 0 0.0595" material="cup_inner" contype="0" conaffinity="0"/>
      <geom name="cup_handle_top" type="capsule" size="0.007" fromto="-0.047 0 0.029 -0.082 0 0.029" material="cup_blue" contype="0" conaffinity="0"/>
      <geom name="cup_handle_side" type="capsule" size="0.007" fromto="-0.082 0 0.029 -0.082 0 -0.029" material="cup_blue" contype="0" conaffinity="0"/>
      <geom name="cup_handle_bottom" type="capsule" size="0.007" fromto="-0.082 0 -0.029 -0.047 0 -0.029" material="cup_blue" contype="0" conaffinity="0"/>
      <geom name="cup_logo" type="ellipsoid" size="0.021 0.0015 0.017" pos="0 -0.050 -0.005" material="cup_logo" contype="0" conaffinity="0"/>
      <geom name="cup_logo_eye_left" type="sphere" size="0.0025" pos="-0.007 -0.052 0.000" material="cup_logo_dark" contype="0" conaffinity="0"/>
      <geom name="cup_logo_eye_right" type="sphere" size="0.0025" pos="0.007 -0.052 0.000" material="cup_logo_dark" contype="0" conaffinity="0"/>
    </body>
    <body name="plate_body" pos="{plate_x:.6f} {plate_y:.6f} {plate_z:.6f}">
      <geom name="plate_rim" type="cylinder" size="0.064 0.005" material="plate_rim" contype="1" conaffinity="1"/>
      <geom name="plate" type="cylinder" size="0.056 0.004" pos="0 0 0.006" material="plate_green" contype="0" conaffinity="0"/>
    </body>
    <body name="carrot_body" pos="{carrot_x:.6f} {carrot_y:.6f} {carrot_z:.6f}" quat="{carrot_quat}">
      <inertial pos="0 0 0" mass="0.05" diaginertia="0.0001 0.0001 0.0001"/>
      <joint name="carrot_joint" type="free"/>
      <geom name="carrot" type="capsule" size="0.025 0.040" material="carrot_orange" contype="0" conaffinity="0"/>
      <geom name="carrot_seam" type="cylinder" size="0.0258 0.0025" material="carrot_seam" contype="0" conaffinity="0"/>
      <geom name="carrot_groove_left" type="cylinder" size="0.0255 0.0008" pos="0 0 -0.020" material="carrot_seam" contype="0" conaffinity="0"/>
      <geom name="carrot_groove_right" type="cylinder" size="0.0255 0.0008" pos="0 0 0.020" material="carrot_seam" contype="0" conaffinity="0"/>
      <geom name="carrot_leaf" type="ellipsoid" size="0.016 0.022 0.009" pos="0 0 -0.067" material="leaf_green" contype="0" conaffinity="0"/>
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

    def _site_xpos(self, physics, site_name: str) -> np.ndarray:
        site_id = physics.model.name2id(site_name, "site")
        return np.asarray(physics.data.site_xpos[site_id], dtype=np.float64).copy()

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
    timeline: Optional[MujocoSourceTimelineResult] = None,
    visual_alignment: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"smoke": smoke.to_dict(), "rollout": rollout.to_dict()}
    if timeline is not None:
        report["timeline"] = timeline.to_dict()
    if visual_alignment is not None:
        report["visual_alignment"] = visual_alignment
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


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
        self._home_qpos = self._qpos.copy()

    def solve(
        self,
        target: Sequence[float],
        *,
        max_iters: int = 36,
        tolerance: float = 0.018,
        damping: float = 0.030,
        posture_weight: float = 0.04,
        posture_target: Optional[Sequence[float]] = None,
        locked_qpos: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        initial_qpos = self._qpos.copy()
        posture = (
            self._home_qpos
            if posture_target is None
            else np.asarray(posture_target, dtype=np.float64)
        )
        require_posture_convergence = posture_target is not None
        if posture.shape != self._home_qpos.shape:
            raise ValueError(
                f"Expected posture target shape {self._home_qpos.shape}, "
                f"got {posture.shape}"
            )
        locked = np.full_like(initial_qpos, np.nan)
        if locked_qpos is not None:
            locked = np.asarray(locked_qpos, dtype=np.float64)
        if locked.shape != self._home_qpos.shape:
            raise ValueError(
                f"Expected locked qpos shape {self._home_qpos.shape}, "
                f"got {locked.shape}"
            )
        locked_mask = np.isfinite(locked)
        free_mask = ~locked_mask
        locked_values = np.clip(
            locked,
            initial_qpos - 0.08,
            initial_qpos + 0.08,
        )
        locked_values = np.clip(locked_values, self._lower, self._upper)
        qpos = initial_qpos.copy()
        qpos[locked_mask] = locked_values[locked_mask]
        for _ in range(max_iters):
            self._physics.data.qpos[self._qpos_addr] = qpos
            self._physics.forward()
            site_pos = np.asarray(
                self._physics.data.site_xpos[self._site_id],
                dtype=np.float64,
            )
            error = target - site_pos
            position_converged = np.linalg.norm(error) <= tolerance
            posture_converged = not require_posture_convergence or bool(
                np.abs(posture[free_mask] - qpos[free_mask]).max(initial=0.0) <= 0.02
            )
            if position_converged and posture_converged:
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
            jac = jacp[:, self._dof_addr[free_mask]]
            lhs = jac @ jac.T + np.eye(3) * damping
            damped_pinv = jac.T @ np.linalg.solve(lhs, np.eye(3))
            delta = damped_pinv @ error
            nullspace = np.eye(int(free_mask.sum())) - damped_pinv @ jac
            delta += posture_weight * nullspace @ (posture[free_mask] - qpos[free_mask])
            qpos[free_mask] = np.clip(
                qpos[free_mask] + np.clip(delta, -0.08, 0.08),
                self._lower[free_mask],
                self._upper[free_mask],
            )

        qpos = np.clip(
            qpos,
            initial_qpos - 0.14,
            initial_qpos + 0.14,
        )
        qpos[locked_mask] = locked_values[locked_mask]
        self._qpos = qpos
        return qpos.copy()


def _source_posture_target(
    state: SceneState,
    config: AlohaCarrotLeftConfig,
) -> Optional[np.ndarray]:
    if len(config.source_qpos) != 6 or len(config.source_ee_pos) != 6:
        return None
    phase_indices = (4, 5) if state.object_phase == "in_cup" else None
    if phase_indices is None:
        return None
    source_ee = np.asarray(config.source_ee_pos, dtype=np.float64)
    reference_idx = min(
        phase_indices,
        key=lambda idx: float(np.linalg.norm(state.ee_pos - source_ee[idx])),
    )
    return np.asarray(config.source_qpos[reference_idx][:6], dtype=np.float64)


def _locked_joint_target(
    state: SceneState,
    config: AlohaCarrotLeftConfig,
) -> Optional[np.ndarray]:
    if state.object_phase == "held" and state.carrot_pos[0] < 0.10:
        wrist_rotate = -0.60
    elif state.object_phase == "in_cup":
        posture = _source_posture_target(state, config)
        wrist_rotate = None if posture is None else float(posture[-1])
    else:
        wrist_rotate = None
    if wrist_rotate is None:
        return None
    target = np.full(6, np.nan, dtype=np.float64)
    target[-1] = wrist_rotate
    return target


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
            camera.set("target", "vx300s_left/camera_focus")
            camera.set("pos", _space(config.wrist_camera_pos))
            camera.set("fovy", f"{config.wrist_camera_fovy:.6f}")
            for attr in (
                "xyaxes",
                "euler",
                "quat",
                "focal",
                "resolution",
                "sensorsize",
            ):
                camera.attrib.pop(attr, None)
    for geom in root.iter("geom"):
        if geom.attrib.get("name", "").startswith("vx300s_left/"):
            geom.set("material", "arm_black")
            geom.attrib.pop("rgba", None)
        if geom.attrib.get("name") in {
            "vx300s_left/1_base",
            "vx300s_left/2_shoulder",
            "vx300s_left/3_upper_arm",
        }:
            geom.set("rgba", "0 0 0 0")
    tree.write(path, encoding="unicode")


def _write_wood_texture(path: Path, seed: int) -> None:
    """Write a deterministic, low-contrast tabletop texture."""

    rng = np.random.default_rng(seed)
    height, width = 512, 1024
    y, x = np.mgrid[:height, :width]

    row_noise = rng.normal(0.0, 1.0, height)
    kernel = np.ones(23, dtype=np.float64) / 23.0
    row_noise = np.convolve(row_noise, kernel, mode="same")
    row_noise /= max(float(np.std(row_noise)), 1e-6)

    warped_y = y + 4.0 * np.sin(x / 91.0) + 1.8 * np.sin(x / 29.0)
    grain = (
        4.0 * np.sin(warped_y / 8.5)
        + 2.2 * np.sin(warped_y / 3.1)
        + 5.0 * row_noise[:, None]
        + 1.6 * np.sin((x + y * 0.7) / 43.0)
    )
    base = np.array([188.0, 188.0, 191.0], dtype=np.float64)
    channel_scale = np.array([1.0, 0.96, 0.92], dtype=np.float64)
    texture = base + grain[..., None] * channel_scale * 1.8

    for center_x, center_y, radius_x, radius_y in (
        (180, 95, 84, 15),
        (730, 330, 110, 18),
    ):
        radius = np.sqrt(
            ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2
        )
        knot = np.exp(-((radius - 1.0) ** 2) * 10.0) * -11.0
        texture += knot[..., None] * channel_scale

    Image.fromarray(np.clip(texture, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _write_cloth_texture(path: Path) -> None:
    """Write broad deterministic folds resembling the green dataset backdrop."""

    y, x = np.mgrid[:512, :512]
    phase = x / 46.0 + 0.65 * np.sin(y / 105.0) + 0.22 * np.sin((x + y) / 37.0)
    folds = 0.88 + 0.075 * np.sin(phase) + 0.035 * np.sin(x / 13.0 + y / 71.0)
    folds -= 0.08 * np.exp(-(((x - 250.0) / 24.0) ** 2))
    grayscale = np.clip(folds * 255.0, 0, 255).astype(np.uint8)
    texture = np.repeat(grayscale[..., None], 3, axis=2)
    Image.fromarray(texture, mode="RGB").save(path)


def _write_source_timeline_contact_sheet(
    path: Path,
    *,
    source_dir: Path,
    primary_images: Sequence[Image.Image],
    wrist_images: Sequence[Image.Image],
) -> None:
    thumb_size = (320, 240)
    rows: List[List[Image.Image]] = []
    for prefix in ("high", "wrist"):
        source_row = []
        for idx in range(1, 7):
            source_path = source_dir / f"{prefix}_{idx:02d}.png"
            if source_path.is_file():
                image = Image.open(source_path).convert("RGB").resize(thumb_size)
            else:
                image = Image.new("RGB", thumb_size, (35, 35, 35))
            _label_image(image, f"source_{prefix}_{idx:02d}")
            source_row.append(image)
        rows.append(source_row)

        rendered = primary_images if prefix == "high" else wrist_images
        rendered_row = []
        for idx, image in enumerate(rendered, start=1):
            image = image.convert("RGB").resize(thumb_size)
            _label_image(image, f"mujoco_{prefix}_{idx:02d}")
            rendered_row.append(image)
        rows.append(rendered_row)

    sheet = Image.new(
        "RGB",
        (thumb_size[0] * 6, thumb_size[1] * len(rows)),
        "white",
    )
    for row_idx, row in enumerate(rows):
        for column_idx, image in enumerate(row):
            sheet.paste(image, (column_idx * thumb_size[0], row_idx * thumb_size[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _label_image(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 170, 20), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))


def _max_rotation_step_degrees(rotations: Sequence[np.ndarray]) -> float:
    max_angle = 0.0
    for previous, current in zip(rotations, rotations[1:]):
        relative = previous.T @ current
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        max_angle = max(max_angle, float(np.degrees(np.arccos(cosine))))
    return max_angle


def _space(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _is_right_arm_body(name: str) -> bool:
    return (
        name == "vx300s_right"
        or name.startswith("vx300s_right/")
        or name == "right"
        or name.startswith("right/")
    )
