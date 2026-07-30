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


def main(_):
    # setup wandb for logging
    wandb.init(name="eval_aloha_pt", project="octo")
    if not FLAGS.finetuned_path:
        raise ValueError("--finetuned_path is required")
    if not 1 <= FLAGS.exec_horizon <= 20:
        raise ValueError("--exec_horizon must be between 1 and 20")
    if FLAGS.video_fps <= 0:
        raise ValueError("--video_fps must be positive")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModelPt.load_pretrained(FLAGS.finetuned_path)['octo_model']
    model.to(device)
    
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

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction, device=device)

        # run rollout for 400 steps
        images = [info["images"]]
        episode_return = 0.0
        while len(images) < FLAGS.max_steps:
            
            obs['timestep_pad_mask'] = obs['timestep_pad_mask'].astype(np.bool_)
            obs = _np2pt(obs, device)
            
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy_fn(tree_map(lambda x: x[None], obs), task)
            actions = actions[0].detach().cpu().numpy()

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            images.extend(info["images"])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")
        video_path = output_dir / f"rollout_{rollout_index:02d}.mp4"
        imageio.mimsave(video_path, images, fps=FLAGS.video_fps)
        logging.info("Saved rollout video to %s", video_path)

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log(
            {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        )
    env.close()


if __name__ == "__main__":
    app.run(main)
