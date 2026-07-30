"""Gym wrapper for the dataset-aligned left-arm ALOHA carrot task."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import gym
except ModuleNotFoundError:
    gym = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import DatasetAlignedController
from octo.sim.aloha_carrot_left import FOLLOWER_GRIPPER_CLOSE
from octo.sim.aloha_carrot_left import FOLLOWER_GRIPPER_OPEN
from octo.sim.aloha_carrot_left import LeftArmCommand


_DEFAULT_IMAGE_KEYS = ("primary", "wrist")
_CAMERA_TO_OBS_KEY = {
    "overhead_cam": "image_primary",
    "wrist_cam_left": "image_wrist",
}


class AlohaGymEnv(gym.Env if gym is not None else object):
    """Single-left-arm carrot pick-and-place environment.

    By default the environment uses the deterministic dataset-aligned controller
    so the left arm autonomously navigates to the carrot and places it in the
    cup. Set ``autonomous=False`` to drive the task with 4D actions:
    ``[target_x, target_y, target_z, follower_gripper]``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(
        self,
        camera_names: Optional[Sequence[str]] = None,
        im_size: int = 256,
        seed: int = 42,
        autonomous: bool = True,
        max_episode_steps: int = 160,
        config: Optional[AlohaCarrotLeftConfig] = None,
    ):
        if gym is None:
            raise ImportError("gym is required to instantiate AlohaGymEnv")

        self.config = config or AlohaCarrotLeftConfig.from_json()
        self.camera_names = tuple(camera_names or self.config.camera_names)
        self._im_size = int(im_size)
        self._rng = np.random.default_rng(seed)
        self._autonomous = bool(autonomous)
        self._max_episode_steps = int(max_episode_steps)
        self._sim = AlohaCarrotLeftSim(self.config, seed=seed)
        self._controller = DatasetAlignedController(self.config)
        self._episode_is_success = 0

        image_spaces = {}
        for camera_name in self.camera_names:
            key = _CAMERA_TO_OBS_KEY[camera_name]
            image_spaces[key] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self._im_size, self._im_size, 3),
                dtype=np.uint8,
            )

        self.observation_space = gym.spaces.Dict(
            {
                **image_spaces,
                "proprio": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(8,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.array(
                [-0.55, -0.05, 0.02, FOLLOWER_GRIPPER_CLOSE],
                dtype=np.float32,
            ),
            high=np.array(
                [0.25, 0.45, 0.40, FOLLOWER_GRIPPER_OPEN],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._controller.reset()
        state = self._sim.reset()
        self._episode_is_success = 0
        obs, raw_images = self._get_obs()
        info = {
            "images": _concat_images(raw_images),
            "raw_images": raw_images,
            "state": state,
            "left_arm_only": True,
            "right_arm_present": False,
        }
        return obs, info

    def step(self, action):
        command = self._select_command(action)
        state = self._sim.step(command)
        obs, raw_images = self._get_obs()
        reward = float(state.reward)
        if self._autonomous:
            terminated = self._controller.is_finished(state)
        else:
            terminated = bool(state.success)
        truncated = bool(state.step_index >= self._max_episode_steps and not terminated)
        if terminated:
            self._episode_is_success = 1
        info = {
            "images": _concat_images(raw_images),
            "raw_images": raw_images,
            "state": state,
            "left_arm_only": True,
            "right_arm_present": False,
            "object_phase": state.object_phase,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._sim.render_primary(self._sim.state)

    def close(self):
        return None

    def get_task(self):
        return {
            "language_instruction": ["pick up the carrot and put it in the cup"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
            "left_arm_only": True,
            "right_arm_present": False,
        }

    def _select_command(self, action) -> LeftArmCommand:
        if self._autonomous or action is None:
            return self._controller.command(self._sim.state)
        array = np.asarray(action, dtype=np.float64).reshape(-1)
        if len(array) < 4:
            raise ValueError("Manual ALOHA carrot actions must have at least 4 values")
        low = self.action_space.low.astype(np.float64)
        high = self.action_space.high.astype(np.float64)
        clipped = np.clip(array[:4], low, high)
        return LeftArmCommand(
            ee_target=(float(clipped[0]), float(clipped[1]), float(clipped[2])),
            gripper=float(clipped[3]),
            speed=0.032,
        )

    def _get_obs(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        raw_images = self._sim.get_observation(render=True)["images"]
        obs: Dict[str, np.ndarray] = {}
        for camera_name in self.camera_names:
            key = _CAMERA_TO_OBS_KEY[camera_name]
            obs[key] = _resize_uint8(raw_images[camera_name], self._im_size)
        obs["proprio"] = self._sim.state.left_qpos.astype(np.float32)
        return obs, raw_images


def _resize_uint8(image: np.ndarray, size: int) -> np.ndarray:
    resized = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
        (size, size),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.uint8)


def _concat_images(images: Dict[str, np.ndarray]) -> np.ndarray:
    ordered = [images[name] for name in ("overhead_cam", "wrist_cam_left") if name in images]
    return np.concatenate(ordered, axis=0)


def _register_env(env_id: str, **kwargs: Any) -> None:
    if gym is None:
        return
    try:
        gym.spec(env_id)
        return
    except Exception:
        pass
    gym.register(env_id, entry_point=lambda: AlohaGymEnv(**kwargs))


_register_env("aloha-carrot-left-v0", autonomous=True)
_register_env("aloha-carrot-left-manual-v0", autonomous=False)
