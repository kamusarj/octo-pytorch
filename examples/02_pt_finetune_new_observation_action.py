"""
This script demonstrates how to finetune OctoPt to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_pt_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=...
"""

import math
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from absl import app, flags, logging
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import tensorflow as tf
import tqdm
import wandb
import numpy as np

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads_pt import L1ActionHeadPt
from octo.model.components.tokenizers_pt import LowdimObsTokenizerPt
from octo.model.octo_model_pt import OctoModelPt, _np2pt
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import process_text
from octo.utils.train_utils_pt import freeze_weights_pt, parameter_groups_pt
from octo.utils.training_metrics import TrainingMetricsRecorder

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        text_processor,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._text_processor = text_processor
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            sample["task"]["language_instruction"] = np.array(
                [sample["task"]["language_instruction"]]
            )
            sample["task"]["pad_mask_dict"]["language_instruction"] = np.array(
                [sample["task"]["pad_mask_dict"]["language_instruction"]]
            )
            sample = process_text(sample, self._text_processor)

            # remove extra dim
            sample["task"]["language_instruction"]["input_ids"] = sample["task"][
                "language_instruction"
            ]["input_ids"][0]
            sample["task"]["language_instruction"]["attention_mask"] = sample["task"][
                "language_instruction"
            ]["attention_mask"][0]

            del sample["dataset_name"]
            sample = _np2pt(sample)
            yield sample


def _to_device(data, device):
    if isinstance(data, dict):
        return {key: _to_device(val, device) for key, val in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("proprio_key", "state", "Proprioceptive observation key.")
flags.DEFINE_string("language_key", "language_instruction", "Language instruction key.")
flags.DEFINE_string("device", "cuda:0", "PyTorch training device.")
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string(
    "dataset_name",
    "aloha_sim_cube_scripted_dataset",
    "TFDS dataset name inside data_dir.",
)
flags.DEFINE_string(
    "primary_image_key",
    "top",
    "Observation key used as Octo's primary RGB image.",
)
flags.DEFINE_string(
    "dataset_statistics",
    None,
    "Optional precomputed Octo statistics JSON; avoids a full RLDS scan.",
)
flags.DEFINE_integer(
    "dataset_subsample_length",
    32,
    "Windows sampled per trajectory before image decoding; lower uses less RAM.",
)
flags.DEFINE_integer(
    "dataset_shuffle_buffer",
    64,
    "Frame shuffle buffer; each item contains decoded image history, so keep this small on low-RAM machines.",
)
flags.DEFINE_integer(
    "action_horizon", 50, "Number of future actions per training window."
)
flags.DEFINE_string(
    "horizon_loss_weights",
    None,
    "Optional comma-separated non-negative loss weights, one per action horizon.",
)
flags.DEFINE_integer(
    "window_size", 2, "Number of observation history frames per training window."
)
flags.DEFINE_integer("num_steps", 5000, "Number of optimizer steps to run.")
flags.DEFINE_integer("save_interval", 1000, "Checkpoint save interval in steps.")
flags.DEFINE_float("learning_rate", 3e-5, "AdamW learning rate.")
flags.DEFINE_integer("warmup_steps", 200, "Linear learning-rate warmup steps.")
flags.DEFINE_float(
    "min_lr_ratio", 0.1, "Final cosine-decay LR as a fraction of peak LR."
)
flags.DEFINE_float("weight_decay", 1e-4, "AdamW weight decay.")
flags.DEFINE_float(
    "max_grad_norm", 1.0, "Clip gradient norm to this value; 0 disables clipping."
)
flags.DEFINE_integer(
    "gradient_accumulation_steps",
    1,
    "Micro-batches averaged for each optimizer step.",
)
flags.DEFINE_integer("log_interval", 10, "Terminal and W&B logging interval.")
flags.DEFINE_integer("plot_interval", 250, "Refresh loss_curve.png interval.")
flags.DEFINE_integer(
    "smoothing_window", 100, "Trailing window shown in the local loss plot."
)
flags.DEFINE_integer("seed", 42, "Random seed for NumPy and PyTorch.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_string("metrics_dir", None, "CSV/PNG directory; defaults to save_dir.")
flags.DEFINE_string(
    "resume_from",
    None,
    "Checkpoint directory to resume from; num_steps remains the final global step.",
)
flags.DEFINE_integer(
    "resume_step", None, "Checkpoint step to resume, or latest when omitted."
)
flags.DEFINE_integer("batch_size", 1, "Batch size for finetuning.")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Legacy alias: freeze the complete transformer (prefer --freeze_policy).",
)
flags.DEFINE_enum(
    "freeze_policy",
    "stage_a",
    ["stage_a", "stage_b", "stage_c", "legacy"],
    "Plan freeze preset: A=new modules only, B=top transformer/vision, C=all except new modules.",
)
flags.DEFINE_float("new_module_learning_rate", None, "Optional LR for new proprio/readout/action modules.")
flags.DEFINE_float("backbone_learning_rate", None, "Optional LR for unfrozen pretrained vision/transformer modules.")


def _freeze_patterns(policy: str) -> list[str]:
    """Return patterns against actual PyTorch names, not stale JAX class names."""
    language = ["*hf_model*", "*language_tokenizer*", "*language_projection*"]
    transformer = ["*octo_transformer*"]
    vision = ["*observation_tokenizers.primary*"]
    if policy == "stage_a":
        return language + transformer + vision
    if policy == "stage_b":
        return language + [
            "*octo_transformer*block_transformer.transformer.encoder_blocks.0.*",
            "*octo_transformer*block_transformer.transformer.encoder_blocks.1.*",
            "*octo_transformer*block_transformer.transformer.encoder_blocks.2.*",
            "*octo_transformer*block_transformer.transformer.encoder_blocks.3.*",
            "*octo_transformer*block_transformer.transformer.encoder_blocks.4.*",
            "*octo_transformer*block_transformer.transformer.encoder_blocks.5.*",
            "*octo_transformer*block_transformer.transformer.encoder_blocks.6.*",
            "*octo_transformer*block_transformer.transformer.encoder_blocks.7.*",
        ]
    if policy == "stage_c":
        return language
    if policy == "legacy":
        return []
    raise ValueError(f"unknown freeze policy: {policy}")


def main(_):
    positive_flags = {
        "num_steps": FLAGS.num_steps,
        "save_interval": FLAGS.save_interval,
        "batch_size": FLAGS.batch_size,
        "gradient_accumulation_steps": FLAGS.gradient_accumulation_steps,
        "log_interval": FLAGS.log_interval,
        "plot_interval": FLAGS.plot_interval,
        "smoothing_window": FLAGS.smoothing_window,
    }
    if any(value < 1 for value in positive_flags.values()):
        raise ValueError(f"These flags must be positive: {positive_flags}")
    if FLAGS.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= FLAGS.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    horizon_loss_weights = None
    if FLAGS.horizon_loss_weights:
        horizon_loss_weights = [
            float(value.strip())
            for value in FLAGS.horizon_loss_weights.split(",")
            if value.strip()
        ]
        if len(horizon_loss_weights) != FLAGS.action_horizon:
            raise ValueError(
                "horizon_loss_weights must contain exactly action_horizon values"
            )
        if any(weight < 0 for weight in horizon_loss_weights) or not any(
            weight > 0 for weight in horizon_loss_weights
        ):
            raise ValueError(
                "horizon_loss_weights must be non-negative with a positive sum"
            )

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)
    # setup wandb for logging
    wandb.init(name="finetune_aloha_pt", project="octo")

    logging.info("Loading pre-trained model...")
    logging.set_verbosity(logging.INFO)

    # load meta information for pretrained model
    meta = OctoModelPt.load_config_and_meta_from_jax(FLAGS.pretrained_path)

    text_processor = meta["text_processor"]
    device = FLAGS.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA training device requested but unavailable: {device}")

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy

    logging.info("Loading finetuning dataset...")
    dataset_kwargs = dict(
        name=FLAGS.dataset_name,
        data_dir=FLAGS.data_dir,
        image_obs_keys={"primary": FLAGS.primary_image_key},
        proprio_obs_key=FLAGS.proprio_key,
        language_key=FLAGS.language_key,
        num_parallel_reads=1,
        num_parallel_calls=1,
    )
    if FLAGS.dataset_statistics:
        dataset_kwargs["dataset_statistics"] = FLAGS.dataset_statistics
    dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=dict(
            window_size=FLAGS.window_size,
            action_horizon=FLAGS.action_horizon,
            subsample_length=FLAGS.dataset_subsample_length,
            num_parallel_calls=1,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
            num_parallel_calls=1,
        ),
        train=True,
    )
    dataset_statistics = dataset.dataset_statistics

    dataset = dataset.repeat().unbatch().shuffle(FLAGS.dataset_shuffle_buffer)

    pytorch_dataset = TorchRLDSDataset(dataset, text_processor)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=FLAGS.batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

    example_batch = next(iter(dataloader))
    proprio_dim = int(example_batch["observation"]["proprio"].shape[-1])
    action_dim = int(example_batch["action"].shape[-1])
    action_mask = example_batch["action_pad_mask"].bool()
    active_action_dims = action_mask.reshape(-1, action_dim).any(dim=0)
    active_action_dim = int(active_action_dims.sum().item())
    if active_action_dim < 1:
        raise ValueError("action loss mask disables every action dimension")
    logging.info(
        "Action contract: %dD; supervised dimensions: %s",
        action_dim,
        torch.nonzero(active_action_dims).flatten().tolist(),
    )

    # modify config --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA

    del meta["config"]["model"]["observation_tokenizers"]["wrist"]
    ###
    meta["config"]["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizerPt,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )

    # LowdimObsTokenizer emits one token per proprio dimension.
    meta["config"]["model"]["num_tokens_dict"] = {
        "primary": 256,
        "language": 16,
        "proprio": proprio_dim,
        "action": 1,
    }

    # Fully override the old action head with a new one (for smaller changes, you can use update_config)
    meta["config"]["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHeadPt,
        input_dim=384,
        action_horizon=FLAGS.action_horizon,
        action_dim=action_dim,
        readout_key="readout_action",
        horizon_loss_weights=horizon_loss_weights,
    )
    meta["example_batch"] = example_batch
    meta["dataset_statistics"] = dataset_statistics

    # initialize new OctoPt model from modified config
    logging.info("Updating model for new observation & action space...")
    model = OctoModelPt.from_config(
        **meta,
        verbose=True,
    )

    # load weights from JAX model
    resume_payload = None
    start_step = 0
    if FLAGS.resume_from:
        resume_dir = Path(FLAGS.resume_from)
        resume_step = FLAGS.resume_step
        if resume_step is None:
            resume_step = max(
                int(path.name)
                for path in resume_dir.iterdir()
                if path.is_dir()
                and path.name.isdigit()
                and (path / "weights.pth").is_file()
            )
        resume_payload = torch.load(
            resume_dir / str(resume_step) / "weights.pth",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(resume_payload["state_dict"])
        start_step = resume_step + 1
        logging.info(
            "Resuming checkpoint %s at optimizer step %d", resume_dir, start_step
        )
    else:
        _, _ = model.load_weights_from_jax(
            FLAGS.pretrained_path, skip_keys_regex=".*hf_model"
        )
    model.to(device)

    # Apply the repository plan's staged policy after model construction and
    # before optimizer creation.  The old ``BlockTransformer_0`` rule matched
    # zero PyTorch parameters; strict matching prevents silently training a
    # component that the run claims to have frozen.
    policy = FLAGS.freeze_policy
    if FLAGS.freeze_transformer and policy == "stage_a":
        policy = "stage_a"
    frozen_keys = list(model.config["optimizer"].get("frozen_keys") or [])
    if policy != "legacy":
        frozen_keys.extend(_freeze_patterns(policy))
    elif FLAGS.freeze_transformer:
        frozen_keys.extend(["*octo_transformer*"])
    freeze_report = freeze_weights_pt(model.module, frozen_keys, strict=bool(frozen_keys))
    logging.info("Resolved freeze policy=%s: %s", policy, freeze_report)

    model.train()

    group_lrs = {}
    new_lr = FLAGS.new_module_learning_rate or FLAGS.learning_rate
    backbone_lr = FLAGS.backbone_learning_rate or FLAGS.learning_rate
    if FLAGS.new_module_learning_rate is not None or FLAGS.backbone_learning_rate is not None:
        group_lrs = {
            "*heads.action*": new_lr,
            "*observation_tokenizers.proprio*": new_lr,
            "*readout*": new_lr,
            "*octo_transformer*": backbone_lr,
            "*observation_tokenizers.primary*": backbone_lr,
        }
    if group_lrs:
        optimizer_params = parameter_groups_pt(
            model.module, group_lrs, default_lr=FLAGS.learning_rate,
            weight_decay=FLAGS.weight_decay,
        )
    else:
        optimizer_params = [param for param in model.module.parameters() if param.requires_grad]
    if not optimizer_params:
        raise RuntimeError("freeze policy left no trainable parameters")
    trainable_params = [param for param in model.module.parameters() if param.requires_grad]
    optimizer = AdamW(optimizer_params, lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    if resume_payload and "optimizer_state_dict" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    # Preserve per-group differential learning rates; only the single-group
    # legacy optimizer receives the global initial LR.
    if not group_lrs:
        for param_group in optimizer.param_groups:
            param_group["lr"] = FLAGS.learning_rate
            param_group["initial_lr"] = FLAGS.learning_rate
    warmup_steps = min(FLAGS.warmup_steps, max(0, FLAGS.num_steps - 1))

    def lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        decay_steps = max(1, FLAGS.num_steps - warmup_steps - 1)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return FLAGS.min_lr_ratio + (1.0 - FLAGS.min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_factor, last_epoch=start_step - 1)

    metrics_dir = Path(FLAGS.metrics_dir or FLAGS.save_dir or "outputs/training")
    metrics = TrainingMetricsRecorder(
        metrics_dir,
        action_dim=active_action_dim,
        smoothing_window=FLAGS.smoothing_window,
        append=start_step > 0,
    )

    example_batch["task"]["pad_mask_dict"]["language_instruction"] = example_batch[
        "task"
    ]["pad_mask_dict"]["language_instruction"][:, 0]

    # run finetuning loop
    logging.info("Starting finetuning...")

    last_step = -1
    dataloader_iter = iter(dataloader)
    if start_step >= FLAGS.num_steps:
        raise ValueError(
            f"resume step {start_step - 1} already reaches num_steps={FLAGS.num_steps}"
        )
    for i in tqdm.trange(start_step, FLAGS.num_steps, dynamic_ncols=True):
        last_step = i
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_mse = 0.0
        accumulated_active_action_dims = 0.0

        for _ in range(FLAGS.gradient_accumulation_steps):
            batch = next(dataloader_iter)
            batch["task"]["pad_mask_dict"]["language_instruction"] = batch["task"][
                "pad_mask_dict"
            ]["language_instruction"][:, 0]
            batch = _to_device(batch, device=device)

            _, head_outputs = model(
                observations=batch["observation"],
                tasks=batch["task"],
                timestep_pad_mask=batch["observation"]["timestep_pad_mask"],
                action_pad_mask=batch["action_pad_mask"],
                gt_actions=batch["action"],
                train=True,
                verbose=False,
                save_attention_mask=True,
            )

            loss = head_outputs["action"][0]
            info = head_outputs["action"][1]
            (loss / FLAGS.gradient_accumulation_steps).backward()
            accumulated_loss += float(info["loss"])
            accumulated_mse += float(info["mse"])
            accumulated_active_action_dims += float(
                info.get("active_action_dims", active_action_dim)
            )

        train_loss = accumulated_loss / FLAGS.gradient_accumulation_steps
        mse = accumulated_mse / FLAGS.gradient_accumulation_steps
        batch_active_action_dims = (
            accumulated_active_action_dims / FLAGS.gradient_accumulation_steps
        )
        if FLAGS.max_grad_norm > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, FLAGS.max_grad_norm
            ).item()
        else:
            gradient_norm = 0.0
        learning_rate = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()

        metric_row = metrics.record(
            step=i,
            train_loss=train_loss,
            mse=mse,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            active_action_dims=batch_active_action_dims,
        )

        if (i + 1) % FLAGS.log_interval == 0 or i == 0:
            logging.info(
                "Step %d; Loss: %.4f; MAE/dim: %.4f; Cur LR = [%.3e]",
                i,
                train_loss,
                metric_row["mae_per_dim"],
                learning_rate,
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "active_action_dims": batch_active_action_dims,
                    "mae_per_dim": metric_row["mae_per_dim"],
                    "mse_per_dim": metric_row["mse_per_dim"],
                    "rmse_per_dim": metric_row["rmse_per_dim"],
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                },
                step=i,
            )

        if (i + 1) % FLAGS.plot_interval == 0:
            metrics.plot()

        if (i + 1) % FLAGS.save_interval == 0:
            # save checkpoint
            model.save_pretrained(
                step=i, checkpoint_path=FLAGS.save_dir, optimizer=optimizer
            )

    if last_step >= 0 and (last_step + 1) % FLAGS.save_interval != 0:
        model.save_pretrained(
            step=last_step,
            checkpoint_path=FLAGS.save_dir,
            optimizer=optimizer,
        )

    if last_step >= 0:
        plot_path = metrics.plot()
        logging.info("Training metrics CSV: %s", metrics.csv_path)
        logging.info("Loss curve PNG: %s", plot_path)
        wandb.log({"loss_curve": wandb.Image(str(plot_path))}, step=last_step)
    metrics.close()
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
