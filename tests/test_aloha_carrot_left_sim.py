from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from octo.sim.aloha_carrot_left import AlohaCarrotLeftConfig
from octo.sim.aloha_carrot_left import AlohaCarrotLeftSim
from octo.sim.aloha_carrot_left import DatasetAlignedController
from octo.sim.aloha_carrot_left import FOLLOWER_GRIPPER_OPEN
from octo.sim.aloha_carrot_left import LeftArmCommand
from octo.sim.aloha_carrot_left import rollout_metrics
from octo.sim.aloha_carrot_left import save_image
from octo.sim.aloha_carrot_left import source_aligned_timeline
from scripts.audit_aloha_carrot_assets import audit_assets
from scripts.audit_aloha_carrot_assets import _ACT_ASSET_NAMES
from scripts.audit_aloha_carrot_assets import _SOURCE_FRAME_NAMES
from scripts.validate_aloha_carrot_left_alignment import validate_alignment
from scripts.validate_aloha_carrot_left_timeline import build_timeline_report
from scripts.validate_aloha_carrot_left_visual_similarity import (
    build_visual_similarity_report,
)


class AlohaCarrotLeftSimTest(unittest.TestCase):
    def setUp(self):
        self.config = AlohaCarrotLeftConfig.from_json()

    def test_config_matches_dataset_aloha_left_arm_contract(self):
        self.assertEqual(self.config.camera_names, ("overhead_cam", "wrist_cam_left"))
        self.assertFalse(self.config.use_right_arm)
        self.assertEqual(len(self.config.left_home_qpos), 8)
        self.assertEqual(tuple(self.config.initial_ee_pos), (-0.36, 0.255, 0.205))
        self.assertEqual(tuple(self.config.cup_landmark_px), (245, 250))
        self.assertEqual(tuple(self.config.plate_landmark_px), (379, 251))
        self.assertEqual(tuple(self.config.carrot_landmark_px), (381, 245))

    def test_old_cube_runtime_identifiers_are_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        forbidden = (
            "from sim_env",
            "BOX_POSE",
            "make_sim_env",
            "aloha-sim-cube-v0",
        )
        offenders = []
        for root_name in ("examples", "scripts", "octo"):
            for path in (repo_root / root_name).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for needle in forbidden:
                    if needle in text:
                        offenders.append(f"{path.relative_to(repo_root)}: {needle}")
        self.assertEqual(offenders, [])

    def test_reset_is_fixed_and_left_arm_only(self):
        sim_a = AlohaCarrotLeftSim(self.config)
        sim_b = AlohaCarrotLeftSim(self.config)
        state_a = sim_a.reset()
        state_b = sim_b.reset()

        np.testing.assert_allclose(state_a.ee_pos, state_b.ee_pos)
        np.testing.assert_allclose(state_a.left_qpos, state_b.left_qpos)
        np.testing.assert_allclose(state_a.carrot_pos, state_b.carrot_pos)
        self.assertEqual(state_a.left_qpos.shape, (8,))
        obs = sim_a.get_observation(render=False)
        self.assertTrue(obs["left_arm_only"])
        self.assertFalse(obs["right_arm_present"])
        self.assertEqual(obs["qpos"].shape, (8,))

    def test_object_does_not_move_or_spin_before_grasp(self):
        sim = AlohaCarrotLeftSim(self.config)
        initial = sim.reset()
        command = LeftArmCommand(
            ee_target=(-0.30, 0.255, 0.22),
            gripper=FOLLOWER_GRIPPER_OPEN,
            speed=0.01,
        )
        for _ in range(24):
            state = sim.step(command)
            self.assertEqual(state.object_phase, "on_plate")
            np.testing.assert_allclose(state.carrot_pos, initial.carrot_pos)
            np.testing.assert_allclose(state.carrot_quat, initial.carrot_quat)

    def test_scripted_controller_succeeds_without_carrot_spin(self):
        sim = AlohaCarrotLeftSim(self.config)
        states = sim.rollout_scripted(controller=DatasetAlignedController(self.config))
        metrics = rollout_metrics(states)

        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["final_state"], "in_cup")
        self.assertEqual(metrics["max_reward"], 4.0)
        self.assertTrue(metrics["left_arm_only"])
        self.assertFalse(metrics["right_arm_present"])
        self.assertEqual(metrics["left_qpos_shape"], [8])
        self.assertLessEqual(metrics["pre_grasp_carrot_translation_std"], 1e-12)
        self.assertLessEqual(metrics["pre_grasp_carrot_quat_std"], 1e-12)
        self.assertLessEqual(metrics["post_place_carrot_translation_std"], 1e-12)
        self.assertLessEqual(metrics["post_place_carrot_quat_std"], 1e-12)
        self.assertLessEqual(metrics["max_carrot_quat_delta"], 1e-12)
        self.assertGreaterEqual(len(states), 55)

    def test_source_aligned_timeline_selects_dataset_like_phases(self):
        sim = AlohaCarrotLeftSim(self.config)
        states = sim.rollout_scripted(controller=DatasetAlignedController(self.config))
        timeline = source_aligned_timeline(states, self.config)

        self.assertEqual([item.frame_name for item in timeline], [f"{i:02d}" for i in range(1, 7)])
        self.assertEqual(
            [item.expected_phase for item in timeline],
            ["on_plate", "on_plate", "held", "held", "in_cup", "in_cup"],
        )
        self.assertEqual([item.state.object_phase for item in timeline], [item.expected_phase for item in timeline])
        self.assertEqual(timeline[0].state_index, 0)
        self.assertEqual(timeline[-1].state_index, len(states) - 1)
        self.assertTrue(
            all(a.state_index <= b.state_index for a, b in zip(timeline, timeline[1:]))
        )
        np.testing.assert_allclose(timeline[1].state.carrot_pos, states[0].carrot_pos)
        self.assertLessEqual(timeline[-1].state.ee_pos[0], -0.48)

    def test_primary_and_wrist_render_shapes(self):
        sim = AlohaCarrotLeftSim(self.config)
        primary = sim.render_primary()
        wrist = sim.render_wrist()
        self.assertEqual(primary.shape, (480, 640, 3))
        self.assertEqual(wrist.shape, (480, 640, 3))
        self.assertGreater(float(primary.std()), 5.0)
        self.assertGreater(float(wrist.std()), 5.0)

    def test_primary_landmark_alignment_report_passes(self):
        report = validate_alignment(self.config)
        self.assertTrue(report["passed"])
        self.assertLessEqual(report["landmark_errors_px"]["cup"], 38.0)
        self.assertLessEqual(report["landmark_errors_px"]["plate"], 30.0)
        self.assertLessEqual(report["landmark_errors_px"]["carrot"], 30.0)
        self.assertTrue(all(item["passed"] for item in report["visual_probes"].values()))

    def test_timeline_report_passes_with_source_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_dir = tmpdir / "source"
            source_dir.mkdir()
            blank = Image.new("RGB", (640, 480), (127, 127, 127))
            for prefix in ("high", "wrist"):
                for idx in range(1, 7):
                    blank.save(source_dir / f"{prefix}_{idx:02d}.png")

            report = build_timeline_report(
                self.config,
                output_dir=tmpdir / "timeline",
                source_dir=source_dir,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["source_summary"]["present_count"], 12)
            self.assertTrue(report["checks"]["wrist_visual_source_like"])
            self.assertTrue(
                all(item["passed"] for item in report["wrist_visual_probes"])
            )
            self.assertTrue((tmpdir / "timeline" / "timeline_contact_sheet.png").is_file())

    def test_visual_similarity_report_passes_with_matched_source_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_dir = tmpdir / "source"
            source_dir.mkdir()
            sim = AlohaCarrotLeftSim(self.config)
            states = sim.rollout_scripted(controller=DatasetAlignedController(self.config))
            for item in source_aligned_timeline(states, self.config):
                save_image(source_dir / f"high_{item.frame_name}.png", sim.render_primary(item.state))
                save_image(source_dir / f"wrist_{item.frame_name}.png", sim.render_wrist(item.state))

            report = build_visual_similarity_report(
                self.config,
                output_dir=tmpdir / "visual_similarity",
                source_dir=source_dir,
            )

            self.assertTrue(report["passed"])
            self.assertGreater(report["summary"]["probe_count"], 0)
            self.assertEqual(
                report["summary"]["probe_count"],
                report["summary"]["passed_probe_count"],
            )
            self.assertTrue(
                (tmpdir / "visual_similarity" / "visual_similarity_contact_sheet.png").is_file()
            )

    def test_asset_audit_distinguishes_current_and_full_validation_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            repo_root = tmpdir / "repo"
            source_dir = repo_root / "source_ep0"
            act_asset_root = tmpdir / "act" / "assets"
            source_dir.mkdir(parents=True)
            act_asset_root.mkdir(parents=True)
            for name in _SOURCE_FRAME_NAMES:
                (source_dir / name).write_bytes(b"source")
            for name in _ACT_ASSET_NAMES:
                (act_asset_root / name).write_bytes(b"asset")

            config = AlohaCarrotLeftConfig.from_mapping(
                {
                    **self.config.to_dict(),
                    "source_reference_dir": str(source_dir.relative_to(repo_root)),
                }
            )
            report = audit_assets(
                repo_root=repo_root,
                config=config,
                act_asset_root=act_asset_root,
            )

            self.assertTrue(report["current_validation_ready"])
            self.assertFalse(report["full_dataset_policy_audit_ready"])
            self.assertIn(
                "data/aloha_carrot_easy/data/chunk-*/*.parquet",
                report["missing_for_full_audit"],
            )
            self.assertIn(
                "checkpoints/octo/full_seed42/49999/weights.pth",
                report["missing_for_full_audit"],
            )
            self.assertIn(
                "checkpoints/act/full_seed42/policy_step_050000.ckpt",
                report["missing_for_full_audit"],
            )

            dataset_dir = repo_root / "data" / "aloha_carrot_easy" / "data" / "chunk-000"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "episode_000.parquet").write_bytes(b"parquet")
            octo_checkpoint = repo_root / "checkpoints" / "octo" / "full_seed42" / "49999"
            octo_checkpoint.mkdir(parents=True)
            for name in (
                "weights.pth",
                "config.json",
                "dataset_statistics.json",
                "example_batch.pickle",
            ):
                (octo_checkpoint / name).write_bytes(b"checkpoint")
            act_checkpoint = (
                repo_root
                / "checkpoints"
                / "act"
                / "full_seed42"
                / "policy_step_050000.ckpt"
            )
            act_checkpoint.parent.mkdir(parents=True)
            act_checkpoint.write_bytes(b"checkpoint")

            full_report = audit_assets(
                repo_root=repo_root,
                config=config,
                act_asset_root=act_asset_root,
            )
            self.assertTrue(full_report["current_validation_ready"])
            self.assertTrue(full_report["full_dataset_policy_audit_ready"])
            self.assertEqual(full_report["missing_for_full_audit"], [])

    def test_mujoco_smoke_and_rollout_when_available(self):
        try:
            from octo.sim.aloha_carrot_mujoco import AlohaCarrotMujocoSmoke
        except Exception as exc:
            self.skipTest(f"MuJoCo backend import unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AlohaCarrotMujocoSmoke(self.config)
            try:
                smoke = runner.run(Path(tmpdir))
                rollout = runner.run_scripted_rollout(Path(tmpdir), render_stride=8)
            except RuntimeError as exc:
                self.skipTest(str(exc))
            self.assertEqual(smoke.primary_shape, (480, 640, 3))
            self.assertEqual(smoke.wrist_shape, (480, 640, 3))
            self.assertTrue(smoke.no_right_arm)
            self.assertLessEqual(smoke.static_object_delta, 1e-9)
            self.assertTrue(rollout.no_right_arm)
            self.assertTrue(rollout.success)
            self.assertEqual(rollout.final_state, "in_cup")
            self.assertLessEqual(rollout.pre_grasp_carrot_translation_std, 1e-12)
            self.assertLessEqual(rollout.post_place_carrot_translation_std, 1e-12)
            self.assertLessEqual(rollout.max_carrot_quat_delta, 1e-12)

    def test_gym_env_uses_left_carrot_task_when_gym_available(self):
        try:
            import gym
            from examples.envs.aloha_sim_env import AlohaGymEnv  # noqa
        except ModuleNotFoundError as exc:
            self.skipTest(f"Gym unavailable: {exc}")

        env = gym.make("aloha-carrot-left-v0")
        obs, info = env.reset()
        self.assertEqual(env.action_space.shape, (4,))
        self.assertEqual(obs["proprio"].shape, (8,))
        self.assertEqual(obs["image_primary"].shape, (256, 256, 3))
        self.assertEqual(obs["image_wrist"].shape, (256, 256, 3))
        self.assertTrue(info["left_arm_only"])
        self.assertFalse(info["right_arm_present"])
        self.assertEqual(
            env.unwrapped.get_task()["language_instruction"],
            ["pick up the carrot and put it in the cup"],
        )

        done = False
        total_reward = 0.0
        for _ in range(80):
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if done or truncated:
                break
        self.assertTrue(done)
        self.assertGreaterEqual(total_reward, 4.0)
        self.assertEqual(info["object_phase"], "in_cup")
        self.assertTrue(env.unwrapped.get_episode_metrics()["success_rate"])
        env.close()


if __name__ == "__main__":
    unittest.main()
