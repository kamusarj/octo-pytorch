"""
This script demonstrates how to load and rollout a finetuned Octo model.
We use the Octo model finetuned on ALOHA sim data from the examples/02_finetune_new_observation_action.py script.

The example environment is the dataset-aligned left-arm ALOHA carrot task
registered by examples/envs/aloha_sim_env.py.

To run this script, run:
    cd examples
    python3 03_pt_eval_finetuned.py --finetuned_path=<checkpoint_directory>
"""
from functools import partial
import json
from pathlib import Path

from absl import app, flags, logging
import gym
import imageio
import numpy as np
import wandb

import torch

# keep this to register ALOHA carrot sim env
from envs.aloha_sim_env import AlohaGymEnv  # noqa

from octo.model.octo_model_pt import OctoModelPt
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_utils_pt import tree_map, _np2pt

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)
flags.DEFINE_integer(
    "exec_horizon",
    4,
    "Number of actions to execute from each predicted 20-action chunk.",
)
flags.DEFINE_integer("max_steps", 160, "Maximum environment steps per rollout.")
flags.DEFINE_integer("num_rollouts", 3, "Number of evaluation rollouts.")
flags.DEFINE_string(
    "output_dir",
    "outputs/eval/aloha_carrot_finetuned",
    "Directory for local simulation rollout videos.",
)
flags.DEFINE_float("video_fps", 12.0, "Local rollout video frame rate.")
flags.DEFINE_enum(
    "wandb_mode",
    "disabled",
    ["disabled", "offline", "online"],
    "W&B logging mode. Disabled requires no API key.",
)


def main(_):
    if not FLAGS.finetuned_path:
        raise ValueError("--finetuned_path is required")
    if FLAGS.exec_horizon < 1:
        raise ValueError("--exec_horizon must be positive")
    if FLAGS.video_fps <= 0:
        raise ValueError("--video_fps must be positive")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(FLAGS.finetuned_path)
    checkpoint_step = _latest_checkpoint_step(checkpoint_root)
    weights_path = checkpoint_root / str(checkpoint_step) / "weights.pth"
    wandb.init(
        name="eval_aloha_pt",
        project="octo",
        mode=FLAGS.wandb_mode,
    )

    # load finetuned model
    logging.info("Loading finetuned model from %s", weights_path)
    model = OctoModelPt.load_pretrained(
        FLAGS.finetuned_path,
        step=checkpoint_step,
    )["octo_model"]
    model.to(device)
    model.eval()
    action_horizon = int(
        model.example_batch["observation"]["task_completed"].shape[-1]
    )
    if FLAGS.exec_horizon > action_horizon:
        raise ValueError(
            f"--exec_horizon={FLAGS.exec_horizon} exceeds checkpoint "
            f"action horizon {action_horizon}"
        )
    
    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_primary": ...
    #     }
    #   }
    ##################################################################################################################
    # This environment consumes the fine-tuning action contract directly:
    # six absolute left-arm joints and one follower-gripper value.
    env = gym.make("aloha-carrot-left-policy-v0")

    # wrap env to normalize proprio
    env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=FLAGS.exec_horizon)

    policy_fn = partial(
        model.sample_actions,
        unnormalization_statistics=model.dataset_statistics["action"],
        generator=torch.Generator(device).manual_seed(0),
    )

    # running rollouts
    for rollout_index in range(FLAGS.num_rollouts):
        obs, info = env.reset()
        policy_timestep = int(info["state"].step_index)

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction, device=device)
        task = _select_checkpoint_task_keys(task, model.example_batch["task"])

        images = [info["images"]]
        episode_return = 0.0
        policy_calls = 0
        predicted_actions = []
        phase_history = [info["state"].object_phase]
        while len(images) < FLAGS.max_steps:
            obs = _complete_policy_observation(
                obs,
                timestep=policy_timestep,
                action_horizon=action_horizon,
            )
            obs = _np2pt(obs, device)
            
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            task = _select_checkpoint_task_keys(task, model.example_batch["task"])
            with torch.inference_mode():
                actions = policy_fn(tree_map(lambda x: x[None], obs), task)
            actions = actions[0].detach().cpu().numpy()
            policy_calls += 1
            predicted_actions.extend(actions[: FLAGS.exec_horizon])

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            images.extend(info["images"])
            policy_timestep = int(info["state"][-1].step_index)
            phase_history.extend(info["object_phase"])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")
        video_path = output_dir / f"rollout_{rollout_index:02d}.mp4"
        imageio.mimsave(video_path, images, fps=FLAGS.video_fps)
        logging.info("Saved rollout video to %s", video_path)
        final_state = info["state"][-1]
        action_array = np.asarray(predicted_actions, dtype=np.float64)
        report = {
            "checkpoint": str(checkpoint_root.resolve()),
            "checkpoint_step": checkpoint_step,
            "weights_path": str(weights_path.resolve()),
            "rollout_index": rollout_index,
            "instruction": language_instruction[0],
            "success": bool(final_state.success),
            "final_phase": final_state.object_phase,
            "episode_return": float(episode_return),
            "steps": int(final_state.step_index),
            "policy_calls": policy_calls,
            "exec_horizon": FLAGS.exec_horizon,
            "action_horizon": action_horizon,
            "action_min": action_array.min(axis=0).tolist(),
            "action_max": action_array.max(axis=0).tolist(),
            "phase_transitions": list(dict.fromkeys(phase_history)),
            "video_path": str(video_path),
            "video_frames": len(images),
        }
        report_path = output_dir / f"rollout_{rollout_index:02d}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log(
            {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        )
    env.close()
    wandb.finish()


def _complete_policy_observation(
    obs,
    *,
    timestep: int,
    action_horizon: int,
):
    obs["proprio"] = np.asarray(obs["proprio"], dtype=np.float32)
    timestep_pad_mask = np.asarray(obs["timestep_pad_mask"], dtype=np.bool_)
    history = int(timestep_pad_mask.shape[0])
    obs["timestep_pad_mask"] = timestep_pad_mask
    obs["timestep"] = np.full((history,), timestep, dtype=np.int32)
    obs["task_completed"] = np.zeros(
        (history, action_horizon),
        dtype=np.bool_,
    )
    obs["pad_mask_dict"] = {
        key: timestep_pad_mask.copy()
        for key in ("image_primary", "image_wrist", "proprio", "timestep")
    }
    return obs


def _select_checkpoint_task_keys(task, template):
    selected = {}
    for key, template_value in template.items():
        if key not in task:
            continue
        value = task[key]
        if isinstance(template_value, dict) and isinstance(value, dict):
            selected[key] = _select_checkpoint_task_keys(value, template_value)
        else:
            selected[key] = value
    return selected


def _latest_checkpoint_step(checkpoint_root: Path) -> int:
    steps = [
        int(path.name)
        for path in checkpoint_root.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / "weights.pth").is_file()
    ]
    if not steps:
        raise FileNotFoundError(
            f"No numbered checkpoint containing weights.pth found in {checkpoint_root}"
        )
    return max(steps)


if __name__ == "__main__":
    app.run(main)
