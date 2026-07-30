"""Dataset-aligned left-arm ALOHA carrot task.

The model in this file is intentionally kinematic.  It is used as the stable
source of task state, object placement, scripted navigation, and fast image
alignment tests.  The optional MuJoCo renderer in ``aloha_carrot_mujoco.py``
uses the same state and ACT meshes when ``dm_control`` is available.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter


HOME_CTRL = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 1.5155])
HOME_QPOS = np.array([0.0, -0.959, 1.182, 0.0, -0.274, 0.0, 1.5155, 1.5155])

FOLLOWER_GRIPPER_OPEN = 1.5155
FOLLOWER_GRIPPER_CLOSE = -0.06135

SIM_GRIPPER_QPOS_OPEN = 0.037
SIM_GRIPPER_QPOS_CLOSE = 0.0078
SIM_GRIPPER_CTRL_OPEN = 0.037
SIM_GRIPPER_CTRL_CLOSE = 0.002

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs/sim/aloha_carrot_left.json"


def _array(values: Sequence[float], *, dtype=np.float64) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _tuple3(values: Sequence[float]) -> Tuple[float, float, float]:
    array = _array(values)
    if array.shape != (3,):
        raise ValueError(f"Expected 3 values, got shape {array.shape}")
    return tuple(float(v) for v in array)


def _tuple4(values: Sequence[float]) -> Tuple[float, float, float, float]:
    array = _array(values)
    if array.shape != (4,):
        raise ValueError(f"Expected 4 values, got shape {array.shape}")
    return tuple(float(v) for v in array)


def _move_toward(current: np.ndarray, target: np.ndarray, max_distance: float) -> np.ndarray:
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance <= max_distance or distance < 1e-12:
        return target.copy()
    return current + delta / distance * max_distance


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def gripper_open_fraction(follower_value: float) -> float:
    span = FOLLOWER_GRIPPER_OPEN - FOLLOWER_GRIPPER_CLOSE
    return _clamp((follower_value - FOLLOWER_GRIPPER_CLOSE) / span, 0.0, 1.0)


def follower_to_sim_gripper(follower_value: float) -> float:
    fraction = gripper_open_fraction(follower_value)
    return SIM_GRIPPER_QPOS_CLOSE + fraction * (
        SIM_GRIPPER_QPOS_OPEN - SIM_GRIPPER_QPOS_CLOSE
    )


@dataclasses.dataclass(frozen=True)
class AlohaCarrotLeftConfig:
    """Fixed reset/camera/object configuration for the carrot task."""

    schema: str = "octo.aloha_carrot_left.dataset_aligned.v1"
    seed: int = 42
    render_size: Tuple[int, int] = (640, 480)
    policy_image_size: Tuple[int, int] = (256, 256)
    wrist_image_size: Tuple[int, int] = (128, 128)
    camera_names: Tuple[str, str] = ("overhead_cam", "wrist_cam_left")
    source_reference_dir: str = (
        "outputs/validation/aloha_carrot_sim_rebuild/source_ep0"
    )
    primary_camera_pos: Tuple[float, float, float] = (0.0, -0.303794, 1.02524)
    primary_camera_quat: Tuple[float, float, float, float] = (
        0.976332,
        0.216277,
        0.0,
        0.0,
    )
    primary_camera_fovy: float = 58.0
    wrist_camera_pos: Tuple[float, float, float] = (
        -0.011,
        -0.0814748,
        -0.0095955,
    )
    wrist_camera_fovy: float = 58.0
    left_home_qpos: Tuple[float, ...] = tuple(float(v) for v in HOME_QPOS)
    left_home_ctrl: Tuple[float, ...] = tuple(float(v) for v in HOME_CTRL)
    initial_ee_pos: Tuple[float, float, float] = (-0.36, 0.255, 0.205)
    mat_center: Tuple[float, float, float] = (-0.035, 0.168, 0.021)
    mat_size: Tuple[float, float] = (0.58, 0.34)
    cup_pos: Tuple[float, float, float] = (-0.145, 0.168, 0.062)
    plate_pos: Tuple[float, float, float] = (0.075, 0.168, 0.038)
    carrot_pos: Tuple[float, float, float] = (0.075, 0.168, 0.067)
    carrot_quat: Tuple[float, float, float, float] = (
        0.70710678,
        0.0,
        0.70710678,
        0.0,
    )
    cup_landmark_px: Tuple[int, int] = (245, 250)
    plate_landmark_px: Tuple[int, int] = (379, 251)
    carrot_landmark_px: Tuple[int, int] = (381, 245)
    use_right_arm: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AlohaCarrotLeftConfig":
        primary = data.get("primary_camera", {})
        wrist = data.get("wrist_camera", {})
        left_arm = data.get("left_arm", {})
        scene = data.get("scene", {})
        return cls(
            schema=str(data.get("schema", cls.schema)),
            seed=int(data.get("seed", cls.seed)),
            render_size=tuple(int(v) for v in data.get("render_size", cls.render_size)),
            policy_image_size=tuple(
                int(v) for v in data.get("policy_image_size", cls.policy_image_size)
            ),
            wrist_image_size=tuple(
                int(v) for v in data.get("wrist_image_size", cls.wrist_image_size)
            ),
            camera_names=tuple(data.get("camera_names", cls.camera_names)),
            source_reference_dir=str(
                data.get("source_reference_dir", cls.source_reference_dir)
            ),
            primary_camera_pos=_tuple3(
                primary.get("position", cls.primary_camera_pos)
            ),
            primary_camera_quat=_tuple4(
                primary.get("quaternion", cls.primary_camera_quat)
            ),
            primary_camera_fovy=float(
                primary.get("fov_y_degrees", cls.primary_camera_fovy)
            ),
            wrist_camera_pos=_tuple3(wrist.get("position", cls.wrist_camera_pos)),
            wrist_camera_fovy=float(
                wrist.get("fov_y_degrees", cls.wrist_camera_fovy)
            ),
            left_home_qpos=tuple(
                float(v) for v in left_arm.get("home_qpos", cls.left_home_qpos)
            ),
            left_home_ctrl=tuple(
                float(v) for v in left_arm.get("home_ctrl", cls.left_home_ctrl)
            ),
            initial_ee_pos=_tuple3(left_arm.get("initial_ee_pos", cls.initial_ee_pos)),
            mat_center=_tuple3(scene.get("mat_center", cls.mat_center)),
            mat_size=tuple(float(v) for v in scene.get("mat_size", cls.mat_size)),
            cup_pos=_tuple3(scene.get("cup_pos", cls.cup_pos)),
            plate_pos=_tuple3(scene.get("plate_pos", cls.plate_pos)),
            carrot_pos=_tuple3(scene.get("carrot_pos", cls.carrot_pos)),
            carrot_quat=_tuple4(scene.get("carrot_quat", cls.carrot_quat)),
            cup_landmark_px=tuple(
                int(v) for v in scene.get("cup_landmark_px", cls.cup_landmark_px)
            ),
            plate_landmark_px=tuple(
                int(v) for v in scene.get("plate_landmark_px", cls.plate_landmark_px)
            ),
            carrot_landmark_px=tuple(
                int(v) for v in scene.get("carrot_landmark_px", cls.carrot_landmark_px)
            ),
            use_right_arm=bool(left_arm.get("use_right_arm", cls.use_right_arm)),
        )

    @classmethod
    def from_json(cls, path: Path | str = _CONFIG_PATH) -> "AlohaCarrotLeftConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_mapping(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "render_size": list(self.render_size),
            "policy_image_size": list(self.policy_image_size),
            "wrist_image_size": list(self.wrist_image_size),
            "camera_names": list(self.camera_names),
            "source_reference_dir": self.source_reference_dir,
            "primary_camera": {
                "name": self.camera_names[0],
                "position": list(self.primary_camera_pos),
                "quaternion": list(self.primary_camera_quat),
                "fov_y_degrees": self.primary_camera_fovy,
            },
            "wrist_camera": {
                "name": self.camera_names[1],
                "position": list(self.wrist_camera_pos),
                "fov_y_degrees": self.wrist_camera_fovy,
            },
            "left_arm": {
                "home_qpos": list(self.left_home_qpos),
                "home_ctrl": list(self.left_home_ctrl),
                "initial_ee_pos": list(self.initial_ee_pos),
                "use_right_arm": self.use_right_arm,
            },
            "scene": {
                "mat_center": list(self.mat_center),
                "mat_size": list(self.mat_size),
                "cup_pos": list(self.cup_pos),
                "plate_pos": list(self.plate_pos),
                "carrot_pos": list(self.carrot_pos),
                "carrot_quat": list(self.carrot_quat),
                "cup_landmark_px": list(self.cup_landmark_px),
                "plate_landmark_px": list(self.plate_landmark_px),
                "carrot_landmark_px": list(self.carrot_landmark_px),
            },
        }


@dataclasses.dataclass(frozen=True)
class LeftArmCommand:
    ee_target: Tuple[float, float, float]
    gripper: float
    speed: float = 0.032


@dataclasses.dataclass(frozen=True)
class SceneState:
    step_index: int
    ee_pos: np.ndarray
    left_qpos: np.ndarray
    gripper: float
    carrot_pos: np.ndarray
    carrot_quat: np.ndarray
    cup_pos: np.ndarray
    plate_pos: np.ndarray
    object_phase: str
    reward: float
    success: bool

    def copy(self) -> "SceneState":
        return SceneState(
            step_index=self.step_index,
            ee_pos=self.ee_pos.copy(),
            left_qpos=self.left_qpos.copy(),
            gripper=float(self.gripper),
            carrot_pos=self.carrot_pos.copy(),
            carrot_quat=self.carrot_quat.copy(),
            cup_pos=self.cup_pos.copy(),
            plate_pos=self.plate_pos.copy(),
            object_phase=self.object_phase,
            reward=float(self.reward),
            success=bool(self.success),
        )


@dataclasses.dataclass(frozen=True)
class TimelineFrame:
    frame_name: str
    label: str
    expected_phase: str
    state_index: int
    state: SceneState


class DatasetAlignedController:
    """Deterministic left-arm controller that grasps the carrot and places it in the cup."""

    def __init__(self, config: Optional[AlohaCarrotLeftConfig] = None):
        self.config = config or AlohaCarrotLeftConfig()
        self.stage = 0

    def reset(self) -> None:
        self.stage = 0

    def _waypoints(self) -> List[Tuple[np.ndarray, float, float]]:
        carrot = _array(self.config.carrot_pos)
        cup = _array(self.config.cup_pos)

        above_carrot = carrot + np.array([0.0, 0.0, 0.128])
        grasp_carrot = carrot + np.array([0.0, 0.0, 0.036])
        lift_carrot = carrot + np.array([0.0, 0.0, 0.18])
        above_cup = cup + np.array([0.0, 0.0, 0.17])
        place_in_cup = cup + np.array([0.0, 0.0, 0.055])
        retreat = np.array([-0.50, 0.32, 0.29])

        return [
            (above_carrot, FOLLOWER_GRIPPER_OPEN, 0.034),
            (grasp_carrot, FOLLOWER_GRIPPER_OPEN, 0.022),
            (grasp_carrot, FOLLOWER_GRIPPER_CLOSE, 0.016),
            (lift_carrot, FOLLOWER_GRIPPER_CLOSE, 0.030),
            (above_cup, FOLLOWER_GRIPPER_CLOSE, 0.034),
            (place_in_cup, FOLLOWER_GRIPPER_CLOSE, 0.020),
            (place_in_cup, FOLLOWER_GRIPPER_OPEN, 0.014),
            (retreat, FOLLOWER_GRIPPER_OPEN, 0.030),
        ]

    def is_finished(self, state: SceneState) -> bool:
        target, gripper, _ = self._waypoints()[-1]
        target_reached = np.linalg.norm(state.ee_pos - target) < 0.008
        gripper_reached = abs(state.gripper - gripper) < 0.035
        return bool(state.success and target_reached and gripper_reached)

    def command(self, state: SceneState) -> LeftArmCommand:
        waypoints = self._waypoints()
        target, gripper, speed = waypoints[min(self.stage, len(waypoints) - 1)]
        target_reached = np.linalg.norm(state.ee_pos - target) < 0.006
        gripper_reached = abs(state.gripper - gripper) < 0.035
        if target_reached and gripper_reached and self.stage < len(waypoints) - 1:
            self.stage += 1
            target, gripper, speed = waypoints[self.stage]

        return LeftArmCommand(
            ee_target=tuple(float(v) for v in target),
            gripper=float(gripper),
            speed=float(speed),
        )


class AlohaCarrotLeftSim:
    """Stable single-left-arm carrot task with dataset-oriented camera rendering."""

    def __init__(
        self,
        config: Optional[AlohaCarrotLeftConfig] = None,
        *,
        seed: Optional[int] = None,
    ):
        self.config = config or AlohaCarrotLeftConfig()
        self.seed = int(self.config.seed if seed is None else seed)
        self._hold_offset = np.zeros(3, dtype=np.float64)
        self._state = self._initial_state()

    @property
    def state(self) -> SceneState:
        return self._state.copy()

    def reset(self) -> SceneState:
        self._hold_offset = np.zeros(3, dtype=np.float64)
        self._state = self._initial_state()
        return self.state

    def _initial_state(self) -> SceneState:
        qpos = _array(self.config.left_home_qpos)
        qpos = qpos.copy()
        qpos[6:] = FOLLOWER_GRIPPER_OPEN
        return SceneState(
            step_index=0,
            ee_pos=_array(self.config.initial_ee_pos),
            left_qpos=qpos,
            gripper=FOLLOWER_GRIPPER_OPEN,
            carrot_pos=_array(self.config.carrot_pos),
            carrot_quat=_array(self.config.carrot_quat),
            cup_pos=_array(self.config.cup_pos),
            plate_pos=_array(self.config.plate_pos),
            object_phase="on_plate",
            reward=0.0,
            success=False,
        )

    def step(self, command: LeftArmCommand) -> SceneState:
        state = self._state
        target = _array(command.ee_target)
        ee_pos = _move_toward(state.ee_pos, target, command.speed)

        gripper_delta = float(command.gripper) - float(state.gripper)
        max_gripper_delta = 0.18
        if abs(gripper_delta) > max_gripper_delta:
            gripper = state.gripper + math.copysign(max_gripper_delta, gripper_delta)
        else:
            gripper = float(command.gripper)

        object_phase = state.object_phase
        carrot_pos = state.carrot_pos.copy()
        carrot_quat = state.carrot_quat.copy()

        close_fraction = 1.0 - gripper_open_fraction(gripper)
        distance_to_carrot = float(np.linalg.norm(ee_pos - carrot_pos))
        if object_phase == "on_plate" and close_fraction > 0.78 and distance_to_carrot < 0.055:
            object_phase = "held"
            self._hold_offset = carrot_pos - ee_pos

        if object_phase == "held":
            carrot_pos = ee_pos + self._hold_offset
            cup_xy_distance = float(np.linalg.norm(carrot_pos[:2] - state.cup_pos[:2]))
            low_over_cup = carrot_pos[2] < state.cup_pos[2] + 0.06
            if gripper_open_fraction(gripper) > 0.62 and cup_xy_distance < 0.05 and low_over_cup:
                object_phase = "in_cup"
                carrot_pos = state.cup_pos + np.array([0.0, 0.0, 0.018])

        if object_phase == "in_cup":
            carrot_pos = state.cup_pos + np.array([0.0, 0.0, 0.018])

        reward = self._reward(object_phase, ee_pos, carrot_pos)
        success = object_phase == "in_cup"
        self._state = SceneState(
            step_index=state.step_index + 1,
            ee_pos=ee_pos,
            left_qpos=self._qpos_from_ee(ee_pos, gripper),
            gripper=gripper,
            carrot_pos=carrot_pos,
            carrot_quat=carrot_quat,
            cup_pos=state.cup_pos.copy(),
            plate_pos=state.plate_pos.copy(),
            object_phase=object_phase,
            reward=reward,
            success=success,
        )
        return self.state

    def rollout_scripted(
        self,
        *,
        max_steps: int = 160,
        controller: Optional[DatasetAlignedController] = None,
        stop_on_success: bool = False,
    ) -> List[SceneState]:
        controller = controller or DatasetAlignedController(self.config)
        controller.reset()
        states = [self.reset()]
        for _ in range(max_steps):
            state = self.step(controller.command(self._state))
            states.append(state)
            if state.success and (stop_on_success or controller.is_finished(state)):
                break
        return states

    def get_observation(self, *, render: bool = True) -> Dict[str, Any]:
        state = self.state
        obs: Dict[str, Any] = {
            "qpos": state.left_qpos.copy(),
            "proprio": state.left_qpos.copy(),
            "left_arm_only": True,
            "right_arm_present": False,
            "object_phase": state.object_phase,
            "reward": state.reward,
        }
        if render:
            obs["images"] = {
                "overhead_cam": self.render_primary(state),
                "wrist_cam_left": self.render_wrist(state),
            }
        return obs

    def primary_landmarks_px(
        self, state: Optional[SceneState] = None
    ) -> Dict[str, Tuple[float, float]]:
        state = state or self._state
        if state.object_phase == "on_plate":
            carrot_px = tuple(float(v) for v in self.config.carrot_landmark_px)
        else:
            carrot_px = self._project_primary(state.carrot_pos)
        return {
            "cup": tuple(float(v) for v in self.config.cup_landmark_px),
            "plate": tuple(float(v) for v in self.config.plate_landmark_px),
            "carrot": carrot_px,
        }

    def render_primary(self, state: Optional[SceneState] = None) -> np.ndarray:
        state = state or self._state
        width, height = self.config.render_size
        image = _draw_room(width, height)
        draw = ImageDraw.Draw(image, "RGBA")

        self._draw_primary_mat(draw)
        self._draw_primary_arm(draw, state)
        self._draw_plate(draw, state)
        if state.object_phase == "in_cup":
            self._draw_carrot(draw, state, camera="primary")
            self._draw_cup(draw, state)
        else:
            self._draw_cup(draw, state)
            self._draw_carrot(draw, state, camera="primary")
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=115, threshold=3))
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def render_wrist(self, state: Optional[SceneState] = None) -> np.ndarray:
        state = state or self._state
        width, height = self.config.render_size
        image = _draw_wrist_room(width, height)
        draw = ImageDraw.Draw(image, "RGBA")

        distance_to_carrot = float(np.linalg.norm(state.ee_pos - state.carrot_pos))
        distance_to_cup = float(np.linalg.norm(state.ee_pos - state.cup_pos))
        if state.object_phase == "on_plate" and distance_to_carrot < 0.09:
            self._draw_wrist_carrot_close(draw, held=False, over_cup=False)
        elif state.object_phase == "held":
            self._draw_wrist_carrot_close(
                draw,
                held=True,
                over_cup=state.carrot_pos[0] < -0.05,
            )
        elif state.object_phase == "in_cup" and distance_to_cup < 0.16:
            self._draw_wrist_place_close(draw)
        else:
            self._draw_wrist_wide(draw, state)

        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _draw_wrist_wide(self, draw: ImageDraw.ImageDraw, state: SceneState) -> None:
        if state.object_phase == "in_cup":
            draw.polygon(
                [(0, 0), (640, 0), (640, 235), (422, 226), (0, 332)],
                fill=(0, 125, 90, 245),
            )
            draw.polygon(
                [(0, 0), (178, 0), (0, 480)],
                fill=(0, 107, 82, 235),
            )
            draw.line((422, 226, 0, 332), fill=(85, 126, 104, 120), width=2)
            mat = [(374, 355), (640, 330), (640, 480), (462, 480)]
            draw.polygon(mat, fill=(244, 86, 139, 106))
            draw.line(mat[:3], fill=(250, 166, 194, 130), width=2)
            draw.ellipse((590, 372, 675, 422), fill=(96, 181, 124, 200))
        else:
            mat = [(410, 255), (640, 244), (640, 480), (505, 480)]
            draw.polygon(mat, fill=(244, 86, 139, 112))
            self._draw_wrist_mat_texture(draw, mat, alpha=42)
            draw.ellipse((552, 330, 690, 454), fill=(86, 211, 232, 230))
            draw.ellipse(
                (572, 340, 653, 398),
                fill=(126, 231, 246, 245),
                outline=(28, 150, 175, 210),
                width=3,
            )
        self._draw_wrist_fingers(draw, mode="wide")

    def _draw_wrist_carrot_close(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        held: bool,
        over_cup: bool,
    ) -> None:
        if over_cup:
            mat = [(156, 214), (640, 125), (640, 480), (100, 480)]
            draw.polygon(mat, fill=(244, 86, 139, 116))
            self._draw_wrist_mat_texture(draw, mat, alpha=58)
            draw.line((105, 205, 640, 113), fill=(229, 232, 218, 120), width=2)
            self._draw_wrist_plate(draw, center=(508, 266), scale=0.86)
            self._draw_wrist_cup(draw, center=(336, 404), scale=1.18)
            self._draw_wrist_vertical_carrot(draw, center_x=332, top=168, bottom=405, scale=1.18)
        else:
            mat = [(155, 0), (640, 0), (640, 480), (136, 480)]
            draw.polygon(mat, fill=(244, 86, 139, 126))
            self._draw_wrist_mat_texture(draw, mat, alpha=72)
            if held:
                self._draw_wrist_plate(draw, center=(390, 306), scale=1.08)
                self._draw_wrist_vertical_carrot(draw, center_x=345, top=138, bottom=398, scale=1.32)
            else:
                self._draw_wrist_plate(draw, center=(358, 191), scale=1.18)
                self._draw_wrist_cup(draw, center=(333, 515), scale=0.95)
                self._draw_wrist_vertical_carrot(draw, center_x=359, top=119, bottom=270, scale=0.92)
        self._draw_wrist_fingers(draw, mode="close")

    def _draw_wrist_place_close(self, draw: ImageDraw.ImageDraw) -> None:
        draw.polygon([(0, 0), (206, 0), (0, 236)], fill=(0, 125, 90, 245))
        draw.line((0, 346, 640, 250), fill=(218, 224, 211, 120), width=2)
        mat = [(372, 292), (640, 252), (640, 480), (406, 480)]
        draw.polygon(mat, fill=(244, 86, 139, 115))
        self._draw_wrist_mat_texture(draw, mat, alpha=48)
        draw.ellipse((596, 355, 690, 411), fill=(91, 174, 120, 220))
        draw.rounded_rectangle((593, 426, 617, 510), radius=11, fill=(239, 126, 40, 215))
        self._draw_wrist_fingers(draw, mode="wide")

    def _draw_wrist_mat_texture(
        self,
        draw: ImageDraw.ImageDraw,
        polygon: Sequence[Tuple[int, int]],
        *,
        alpha: int,
    ) -> None:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        x1, x2 = max(0, min(xs)), min(640, max(xs))
        y1, y2 = max(0, min(ys)), min(480, max(ys))
        for x in range(x1 - 70, x2 + 80, 12):
            draw.line((x, y1, x + 84, y2), fill=(255, 174, 205, alpha), width=1)
        for x in range(x1 - 20, x2 + 30, 18):
            draw.line((x, y1, x + 35, y2), fill=(200, 45, 108, max(20, alpha // 2)), width=1)

    def _draw_wrist_plate(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        center: Tuple[int, int],
        scale: float,
    ) -> None:
        x, y = center
        rx = int(round(72 * scale))
        ry = int(round(50 * scale))
        draw.ellipse((x - rx - 5, y - ry + 10, x + rx + 8, y + ry + 20), fill=(0, 0, 0, 34))
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=(85, 172, 122, 244))
        draw.ellipse(
            (x - int(rx * 0.66), y - int(ry * 0.58), x + int(rx * 0.68), y + int(ry * 0.56)),
            fill=(127, 196, 145, 246),
            outline=(203, 248, 215, 70),
            width=2,
        )
        draw.arc((x - rx + 13, y - ry + 7, x + rx - 13, y + ry - 7), 190, 330, fill=(230, 255, 235, 120), width=3)

    def _draw_wrist_cup(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        center: Tuple[int, int],
        scale: float,
    ) -> None:
        x, y = center
        rx = int(round(62 * scale))
        ry = int(round(52 * scale))
        draw.ellipse((x - rx, y - ry, x + rx, y + ry + 58), fill=(86, 211, 232, 225))
        draw.ellipse(
            (x - int(rx * 0.78), y - int(ry * 0.75), x + int(rx * 0.78), y + int(ry * 0.25)),
            fill=(126, 231, 246, 240),
            outline=(34, 154, 178, 210),
            width=3,
        )
        draw.ellipse(
            (x - int(rx * 0.48), y - int(ry * 0.48), x + int(rx * 0.48), y + int(ry * 0.02)),
            fill=(86, 184, 211, 230),
        )
        draw.rectangle((x - int(rx * 0.32), y - int(ry * 0.1), x + int(rx * 0.24), y + int(ry * 0.02)), fill=(233, 238, 240, 130))

    def _draw_wrist_vertical_carrot(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        center_x: int,
        top: int,
        bottom: int,
        scale: float,
    ) -> None:
        width = int(round(42 * scale))
        cap = int(round(34 * scale))
        draw.ellipse((center_x - width, bottom - 9, center_x + width, bottom + 16), fill=(0, 0, 0, 34))
        draw.rounded_rectangle(
            (center_x - width, top + cap // 2, center_x + width, bottom - cap),
            radius=max(12, width // 2),
            fill=(239, 126, 40, 255),
            outline=(173, 80, 23, 210),
            width=2,
        )
        draw.pieslice(
            (center_x - width, top, center_x + width, top + cap * 2),
            180,
            360,
            fill=(255, 151, 58, 255),
            outline=(173, 80, 23, 210),
        )
        draw.ellipse(
            (center_x - width, bottom - cap - 10, center_x + width, bottom - cap + 18),
            fill=(222, 103, 32, 245),
            outline=(148, 72, 28, 160),
            width=1,
        )
        draw.ellipse(
            (
                center_x - int(width * 0.78),
                bottom - cap + 8,
                center_x + int(width * 0.78),
                bottom + int(cap * 0.55),
            ),
            fill=(48, 143, 106, 245),
        )
        draw.line(
            (center_x + int(width * 0.32), top + 16, center_x + int(width * 0.45), bottom - cap),
            fill=(255, 226, 151, 160),
            width=max(3, int(5 * scale)),
        )
        seam_y = bottom - cap - 8
        draw.line((center_x - width + 2, seam_y, center_x + width - 2, seam_y), fill=(154, 71, 29, 145), width=2)
        for y in range(top + 42, bottom - cap - 14, max(24, int(30 * scale))):
            draw.line((center_x - width + 8, y, center_x + width - 11, y - 9), fill=(255, 197, 97, 130), width=2)

    def _draw_wrist_fingers(self, draw: ImageDraw.ImageDraw, *, mode: str) -> None:
        if mode == "close":
            left_finger = [(0, 480), (0, 412), (207, 280), (268, 313), (178, 480)]
            right_finger = [(640, 480), (548, 480), (414, 314), (476, 287), (640, 421)]
            left_line = (74, 416, 218, 302)
            right_line = (468, 307, 593, 430)
        else:
            left_finger = [(0, 480), (0, 364), (184, 274), (224, 315), (119, 480)]
            right_finger = [(640, 480), (528, 480), (421, 314), (458, 282), (640, 388)]
            left_line = (58, 388, 190, 299)
            right_line = (455, 304, 578, 398)
        draw.polygon(left_finger, fill=(7, 8, 8, 255))
        draw.polygon(right_finger, fill=(7, 8, 8, 255))
        draw.line(left_line, fill=(56, 59, 58, 220), width=3)
        draw.line(right_line, fill=(56, 59, 58, 220), width=3)
        draw.line((left_line[0] + 15, left_line[1] - 34, left_line[2] - 8, left_line[3] + 12), fill=(86, 90, 87, 90), width=6)
        draw.line((right_line[0] + 6, right_line[1] + 10, right_line[2] - 12, right_line[3] - 24), fill=(86, 90, 87, 90), width=6)

    def scene_metadata(self) -> Dict[str, Any]:
        return {
            "schema": self.config.schema,
            "camera_names": list(self.config.camera_names),
            "left_arm_only": True,
            "right_arm_present": False,
            "left_qpos_shape": [8],
            "object_reset": {
                "cup_pos": list(self.config.cup_pos),
                "plate_pos": list(self.config.plate_pos),
                "carrot_pos": list(self.config.carrot_pos),
                "carrot_quat": list(self.config.carrot_quat),
            },
            "reference": {
                "aloha_sim": "https://github.com/google-deepmind/aloha_sim",
                "source_reference_dir": self.config.source_reference_dir,
            },
        }

    def _reward(self, object_phase: str, ee_pos: np.ndarray, carrot_pos: np.ndarray) -> float:
        if object_phase == "in_cup":
            return 4.0
        if object_phase == "held":
            return 2.0
        if np.linalg.norm(ee_pos - carrot_pos) < 0.075:
            return 1.0
        return 0.0

    def _qpos_from_ee(self, ee_pos: np.ndarray, gripper: float) -> np.ndarray:
        qpos = np.array(self.config.left_home_qpos, dtype=np.float64)
        dx = float(ee_pos[0] - self.config.initial_ee_pos[0])
        dy = float(ee_pos[1] - self.config.initial_ee_pos[1])
        dz = float(ee_pos[2] - self.config.initial_ee_pos[2])
        qpos[:6] = [
            _clamp(-0.35 + dx * 1.55 - dy * 0.25, -1.45, 1.45),
            _clamp(-0.96 + dz * 3.1 + dy * 0.55, -1.55, 0.65),
            _clamp(1.16 - dz * 2.2 + dx * 0.38, -0.4, 1.55),
            _clamp(-0.18 + dx * 0.75, -1.8, 1.8),
            _clamp(-0.30 - dz * 1.4, -1.3, 1.1),
            _clamp(0.04 + dx * 0.25, -1.6, 1.6),
        ]
        qpos[6:] = gripper
        return qpos

    def _project_primary(self, point: Sequence[float]) -> Tuple[float, float]:
        x, y, z = (float(v) for v in point)
        px = 333.5 + x * 610.0 + (y - 0.168) * 22.0
        py = 251.0 - (y - 0.168) * 330.0 - (z - 0.038) * 190.0
        return float(px), float(py)

    def _draw_primary_mat(self, draw: ImageDraw.ImageDraw) -> None:
        mat = [(169, 223), (473, 223), (516, 326), (138, 327)]
        draw.polygon(mat, fill=(247, 83, 136, 116))
        draw.line(mat + [mat[0]], fill=(255, 162, 193, 135), width=2)
        draw.line((198, 244, 489, 245), fill=(228, 51, 116, 65), width=1)
        for x in range(230, 501, 27):
            draw.line((x, 225, x + 28, 326), fill=(234, 65, 124, 38), width=1)

    def _draw_primary_arm(self, draw: ImageDraw.ImageDraw, state: SceneState) -> None:
        ee_px = self._project_primary(state.ee_pos)
        base = (-45, 18)
        elbow = (max(10, ee_px[0] - 128), max(10, ee_px[1] - 95))
        wrist = (ee_px[0] - 16, ee_px[1] - 18)
        draw.line((base[0], base[1], elbow[0], elbow[1]), fill=(8, 9, 9, 255), width=22)
        draw.line((elbow[0], elbow[1], wrist[0], wrist[1]), fill=(10, 10, 10, 255), width=18)
        open_fraction = gripper_open_fraction(state.gripper)
        spread = 10 + 24 * open_fraction
        tip = (ee_px[0], ee_px[1])
        draw.line((wrist[0], wrist[1], tip[0] - spread, tip[1] + 16), fill=(0, 0, 0, 255), width=8)
        draw.line((wrist[0], wrist[1], tip[0] + spread, tip[1] + 16), fill=(0, 0, 0, 255), width=8)

    def _draw_plate(self, draw: ImageDraw.ImageDraw, state: SceneState) -> None:
        center = self.config.plate_landmark_px
        x, y = center
        shadow = (x - 49, y + 8, x + 51, y + 30)
        draw.ellipse(shadow, fill=(0, 0, 0, 35))
        draw.ellipse((x - 43, y - 14, x + 44, y + 17), fill=(86, 167, 119, 255), outline=(31, 96, 74, 230), width=2)
        draw.ellipse((x - 30, y - 9, x + 31, y + 11), fill=(125, 193, 142, 255))

    def _draw_cup(self, draw: ImageDraw.ImageDraw, state: SceneState) -> None:
        x, y = self.config.cup_landmark_px
        draw.ellipse((x - 52, y + 30, x + 42, y + 47), fill=(0, 0, 0, 36))
        draw.rounded_rectangle((x - 35, y - 20, x + 32, y + 55), radius=12, fill=(82, 206, 231, 255), outline=(33, 155, 184, 220), width=2)
        draw.ellipse((x - 34, y - 30, x + 33, y + 4), fill=(100, 224, 242, 255), outline=(31, 150, 176, 235), width=3)
        draw.ellipse((x - 22, y - 22, x + 22, y - 3), fill=(82, 183, 210, 255))
        draw.rounded_rectangle((x - 57, y - 3, x - 32, y + 31), radius=8, outline=(41, 171, 202, 245), width=7)
        draw.ellipse((x - 4, y + 12, x + 16, y + 29), fill=(62, 153, 209, 140))
        draw.rectangle((x - 20, y + 17, x + 14, y + 20), fill=(233, 238, 240, 125))

    def _draw_carrot(
        self,
        draw: ImageDraw.ImageDraw,
        state: SceneState,
        *,
        camera: str,
    ) -> None:
        if camera == "primary" and state.object_phase == "on_plate":
            x, y = self.config.carrot_landmark_px
        elif camera == "primary" and state.object_phase == "in_cup":
            x, y = self.config.cup_landmark_px
            draw.ellipse((x - 10, y - 20, x + 12, y - 8), fill=(239, 126, 40, 210))
            return
        else:
            x, y = self._project_primary(state.carrot_pos)
        x = int(round(x))
        y = int(round(y))
        draw.ellipse((x - 39, y + 9, x + 42, y + 20), fill=(0, 0, 0, 36))
        body = [(x - 26, y - 12), (x + 22, y - 12), (x + 38, y), (x + 22, y + 13), (x - 26, y + 13)]
        draw.polygon(body, fill=(239, 126, 40, 255), outline=(179, 84, 24, 230))
        draw.ellipse((x - 32, y - 12, x - 7, y + 13), fill=(255, 153, 62, 255), outline=(179, 84, 24, 210), width=1)
        for offset in (-12, 2, 15):
            draw.line((x + offset, y - 10, x + offset + 4, y + 11), fill=(255, 196, 95, 150), width=3)
        draw.polygon(
            [(x - 39, y - 4), (x - 51, y - 13), (x - 43, y + 1), (x - 53, y + 10), (x - 37, y + 5)],
            fill=(77, 151, 72, 245),
        )


def rollout_metrics(states: Sequence[SceneState]) -> Dict[str, Any]:
    if not states:
        raise ValueError("states must not be empty")
    carrot_positions = np.stack([s.carrot_pos for s in states])
    carrot_quats = np.stack([s.carrot_quat for s in states])
    pre_grasp_positions = np.stack(
        [s.carrot_pos for s in states if s.object_phase == "on_plate"]
    )
    pre_grasp_quats = np.stack(
        [s.carrot_quat for s in states if s.object_phase == "on_plate"]
    )
    post_place_states = [s for s in states if s.object_phase == "in_cup"]
    post_place_positions = np.stack(
        [s.carrot_pos for s in post_place_states] or [states[-1].carrot_pos]
    )
    post_place_quats = np.stack(
        [s.carrot_quat for s in post_place_states] or [states[-1].carrot_quat]
    )
    return {
        "success": bool(states[-1].success),
        "max_reward": float(max(s.reward for s in states)),
        "steps": int(states[-1].step_index),
        "final_state": states[-1].object_phase,
        "left_arm_only": True,
        "right_arm_present": False,
        "left_qpos_shape": list(states[-1].left_qpos.shape),
        "pre_grasp_carrot_translation_std": float(pre_grasp_positions.std(axis=0).max()),
        "pre_grasp_carrot_quat_std": float(pre_grasp_quats.std(axis=0).max()),
        "post_place_carrot_translation_std": float(
            post_place_positions.std(axis=0).max()
        ),
        "post_place_carrot_quat_std": float(post_place_quats.std(axis=0).max()),
        "max_carrot_step_translation": float(
            np.linalg.norm(np.diff(carrot_positions, axis=0), axis=1).max()
            if len(states) > 1
            else 0.0
        ),
        "max_carrot_quat_delta": float(
            np.linalg.norm(np.diff(carrot_quats, axis=0), axis=1).max()
            if len(states) > 1
            else 0.0
        ),
    }


def source_aligned_timeline(
    states: Sequence[SceneState],
    config: Optional[AlohaCarrotLeftConfig] = None,
) -> List[TimelineFrame]:
    """Select six semantic rollout frames matching the available source frames."""

    if not states:
        raise ValueError("states must not be empty")
    config = config or AlohaCarrotLeftConfig()
    on_plate = [idx for idx, state in enumerate(states) if state.object_phase == "on_plate"]
    held = [idx for idx, state in enumerate(states) if state.object_phase == "held"]
    in_cup = [idx for idx, state in enumerate(states) if state.object_phase == "in_cup"]
    if not on_plate or not held or not in_cup:
        raise ValueError(
            "timeline requires on_plate, held, and in_cup phases; "
            f"got phases={sorted({state.object_phase for state in states})}"
        )

    grasp_target = _array(config.carrot_pos) + np.array([0.0, 0.0, 0.036])
    approach_candidates = on_plate[1:] or on_plate
    approach_idx = min(
        approach_candidates,
        key=lambda idx: float(np.linalg.norm(states[idx].ee_pos - grasp_target)),
    )
    first_held_idx = held[min(len(held) - 1, max(1, int(round(len(held) * 0.20))))]
    transfer_idx = held[min(len(held) - 1, max(0, int(round(len(held) * 0.72))))]
    place_idx = in_cup[0]
    retreat_idx = len(states) - 1
    selections = [
        ("01", "reset", "on_plate", 0),
        ("02", "approach_carrot", "on_plate", approach_idx),
        ("03", "grasp_lift", "held", first_held_idx),
        ("04", "transfer_to_cup", "held", transfer_idx),
        ("05", "placed_in_cup", "in_cup", place_idx),
        ("06", "retreat", "in_cup", retreat_idx),
    ]
    return [
        TimelineFrame(
            frame_name=frame_name,
            label=label,
            expected_phase=expected_phase,
            state_index=state_index,
            state=states[state_index],
        )
        for frame_name, label, expected_phase, state_index in selections
    ]


def save_image(path: Path | str, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_gif(path: Path | str, frames: Iterable[np.ndarray], *, duration_ms: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    if not pil_frames:
        raise ValueError("frames must not be empty")
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _draw_room(width: int, height: int) -> Image.Image:
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    table = np.zeros((height, width, 3), dtype=np.uint8)
    table[..., 0] = np.clip(203 + 14 * np.sin(x / 25) + 9 * np.sin((x + y) / 41), 0, 255)
    table[..., 1] = np.clip(188 + 9 * np.sin(x / 28) + 5 * np.sin(y / 19), 0, 255)
    table[..., 2] = np.clip(166 + 7 * np.sin(x / 18), 0, 255)

    image = Image.fromarray(table, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.polygon([(0, 0), (640, 0), (640, 120), (530, 92), (112, 93), (0, 132)], fill=(0, 125, 90, 255))
    draw.polygon([(0, 0), (128, 0), (0, 135)], fill=(0, 105, 78, 235))
    draw.polygon([(512, 0), (640, 0), (640, 120)], fill=(0, 105, 78, 210))
    draw.line((112, 93, 530, 92), fill=(133, 137, 117, 120), width=2)
    draw.ellipse((164, 101, 185, 108), fill=(38, 114, 123, 180))
    draw.rounded_rectangle((8, 291, 58, 303), radius=2, fill=(204, 235, 237, 120))
    draw.rounded_rectangle((62, 295, 84, 303), radius=2, fill=(213, 242, 238, 135))
    draw.rounded_rectangle((607, 173, 640, 186), radius=2, fill=(210, 240, 245, 140))
    return image


def _draw_wrist_room(width: int, height: int) -> Image.Image:
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    table = np.zeros((height, width, 3), dtype=np.uint8)
    table[..., 0] = np.clip(206 + 16 * np.sin(x / 34) + 7 * np.sin((x + y) / 55), 0, 255)
    table[..., 1] = np.clip(193 + 8 * np.sin(x / 41) + 4 * np.sin(y / 29), 0, 255)
    table[..., 2] = np.clip(174 + 6 * np.sin(x / 25), 0, 255)
    image = Image.fromarray(table, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.polygon([(0, 0), (226, 0), (0, 315)], fill=(0, 125, 90, 255))
    draw.line((226, 0, 0, 315), fill=(85, 126, 104, 125), width=2)
    draw.ellipse((203, 69, 233, 82), fill=(52, 117, 122, 160))
    draw.rounded_rectangle((95, 271, 139, 285), radius=2, fill=(205, 238, 242, 120))
    draw.rounded_rectangle((143, 277, 174, 287), radius=2, fill=(214, 244, 241, 135))
    return image
