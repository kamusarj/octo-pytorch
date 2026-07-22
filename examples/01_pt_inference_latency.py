"""Benchmark Octo PyTorch inference latency on a CUDA GPU.

The output schema is intentionally backend-neutral so a future vla.cpp runner
can emit the same metric names for direct comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
import json
import logging
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# JAX is only needed to read the original checkpoint. Keep it off the GPU so
# PyTorch owns the CUDA memory used for inference.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers\..*")

import numpy as np
import torch

from octo.model.octo_model_pt import OctoModelPt
from octo.utils.latency import LatencyTracker, format_latency_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Octo PyTorch policy latency on CUDA."
    )
    parser.add_argument(
        "--checkpoint",
        default="hf://rail-berkeley/octo-small-1.5",
        help="Original JAX/Hugging Face checkpoint or a fine-tuned PyTorch checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-format",
        choices=("auto", "jax", "pytorch"),
        default="auto",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Exact PyTorch checkpoint step; defaults to latest.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pipeline-fingerprint", default=None)
    parser.add_argument("--dataset", default="bridge_dataset")
    parser.add_argument(
        "--input-source",
        choices=("rlds", "aloha-hf", "golden"),
        default="rlds",
        help="Dataset input source used when --image is not passed.",
    )
    parser.add_argument(
        "--sample-dataset",
        type=Path,
        default=Path("tests/debug_dataset/bridge_dataset/1.0.0"),
        help="TFDS/RLDS directory used when --image is not passed.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--aloha-repo-id",
        default="lerobot/aloha_sim_transfer_cube_scripted",
        help="Hugging Face LeRobot dataset used by --input-source aloha-hf.",
    )
    parser.add_argument(
        "--aloha-video",
        type=Path,
        default=Path(
            "data/aloha_sim_transfer_cube_scripted/videos/"
            "observation.images.top/chunk-000/file-000.mp4"
        ),
        help="Local top-camera MP4 for the ALOHA sample.",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("golden_samples/aloha_static_fork_pick_up_val20"),
        help=(
            "Golden sample set directory used with --input-source golden. "
            "--episode-index selects sample_NNN from its manifest."
        ),
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--step-index",
        type=int,
        default=-1,
        help="Zero-based step index; negative values count from the end (default: -1).",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the sample's language instruction.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional RGB image. It is resized to 256x256 before benchmarking.",
    )
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--profile-modules",
        action="store_true",
        help=(
            "Run a separate CUDA-event profile for major stages and curated "
            "model submodules. Profile samples are not mixed into total latency."
        ),
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=3,
        help="Number of separate module-profile iterations (default: 3).",
    )
    parser.add_argument(
        "--show-model-warnings",
        action="store_true",
        help="Keep repeated missing optional observation warnings in benchmark logs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/inference_latency/latest.json"),
        help="Machine-readable result path (default: outputs/inference_latency/latest.json).",
    )
    args = parser.parse_args()
    if (
        args.window_size < 1
        or args.warmup < 0
        or args.runs < 1
        or args.profile_runs < 1
    ):
        parser.error(
            "window-size, runs, and profile-runs must be positive; "
            "warmup cannot be negative"
        )
    if args.episode_index < 0:
        parser.error("episode-index must be non-negative")
    return args


def checkpoint_format(path: str, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    if path.startswith("hf://"):
        return "jax"
    return "pytorch" if (Path(path) / "example_batch.pickle").exists() else "jax"


def load_model(path: str, fmt: str, step: Optional[int] = None) -> OctoModelPt:
    if fmt == "pytorch":
        return OctoModelPt.load_pretrained(path, step=step)["octo_model"]
    if step is not None:
        raise ValueError("--checkpoint-step is only supported for PyTorch checkpoints")
    return OctoModelPt.load_pretrained_from_jax(path, skip_keys_regex=".*hf_model")[
        "octo_model"
    ]


def resolve_local_checkpoint_step(path: str, requested: Optional[int]) -> Optional[int]:
    """Return the concrete step used for a local PyTorch checkpoint."""
    if not Path(path).is_dir():
        return requested
    if requested is not None:
        if not (Path(path) / str(requested) / "weights.pth").is_file():
            raise FileNotFoundError(f"Checkpoint weights not found for step {requested}")
        return int(requested)
    steps = [
        int(p.name) for p in Path(path).iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "weights.pth").is_file()
    ]
    return max(steps) if steps else None


def measure_cuda_call(
    fn: Callable[[], torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, float, float]:
    """Return function output, synchronized wall latency, and CUDA-event latency."""

    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start_event.record()
    output = fn()
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
    gpu_ms = start_event.elapsed_time(end_event)
    return output, wall_ms, gpu_ms


class CudaProfileRecorder:
    """Collect nested CUDA-event ranges without synchronizing every submodule."""

    def __init__(self, device: torch.device):
        self.device = device
        self._current: Optional[Dict[str, list]] = None
        self._iterations: list[Dict[str, list]] = []
        self.module_metadata: Dict[str, Dict[str, str]] = {}

    def begin_iteration(self) -> None:
        if self._current is not None:
            raise RuntimeError("A profile iteration is already active")
        self._current = {}

    def _begin(self, category: str, name: str) -> Dict[str, Any]:
        if self._current is None:
            raise RuntimeError("begin_iteration() must be called before profiling")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        token = {
            "category": category,
            "name": name,
            "start": start,
            "end": end,
            "host_start_ns": time.perf_counter_ns(),
        }
        start.record()
        return token

    def _end(self, token: Dict[str, Any]) -> None:
        token["end"].record()
        token["host_enqueue_ms"] = (
            time.perf_counter_ns() - token["host_start_ns"]
        ) / 1_000_000.0
        key = f"{token['category']}:{token['name']}"
        assert self._current is not None
        self._current.setdefault(key, []).append(token)

    @contextmanager
    def range(self, name: str):
        token = self._begin("stage", name)
        try:
            yield
        finally:
            self._end(token)

    def begin_module(self, name: str) -> Dict[str, Any]:
        return self._begin("module", name)

    def end_module(self, token: Dict[str, Any]) -> None:
        self._end(token)

    def end_iteration(self) -> None:
        if self._current is None:
            raise RuntimeError("No active profile iteration")
        torch.cuda.synchronize(self.device)
        resolved: Dict[str, list] = {}
        for key, tokens in self._current.items():
            resolved[key] = [
                {
                    "cuda_ms": token["start"].elapsed_time(token["end"]),
                    "host_enqueue_ms": token["host_enqueue_ms"],
                }
                for token in tokens
            ]
        self._iterations.append(resolved)
        self._current = None

    @staticmethod
    def _summary(name: str, values: list[float]) -> Dict[str, Any]:
        tracker = LatencyTracker(name)
        for value in values:
            tracker.record(value)
        return tracker.summary()

    def report(self) -> Dict[str, Any]:
        all_keys = sorted({key for iteration in self._iterations for key in iteration})
        groups: Dict[str, Dict[str, Any]] = {"stages": {}, "modules": {}}
        for key in all_keys:
            category, name = key.split(":", 1)
            calls_by_iteration = [
                iteration.get(key, []) for iteration in self._iterations
            ]
            cuda_per_call = [
                call["cuda_ms"] for calls in calls_by_iteration for call in calls
            ]
            host_per_call = [
                call["host_enqueue_ms"]
                for calls in calls_by_iteration
                for call in calls
            ]
            cuda_per_inference = [
                sum(call["cuda_ms"] for call in calls) for calls in calls_by_iteration
            ]
            calls_per_inference = [len(calls) for calls in calls_by_iteration]
            item = {
                "inclusive": True,
                "calls_total": sum(calls_per_inference),
                "calls_per_inference": {
                    "mean": sum(calls_per_inference) / len(calls_per_inference),
                    "min": min(calls_per_inference),
                    "max": max(calls_per_inference),
                },
                "cuda_per_call": self._summary(f"{name}.cuda_per_call", cuda_per_call),
                "cuda_per_inference": self._summary(
                    f"{name}.cuda_per_inference", cuda_per_inference
                ),
                "host_enqueue_per_call": self._summary(
                    f"{name}.host_enqueue_per_call", host_per_call
                ),
            }
            if category == "module":
                item.update(self.module_metadata[name])
                groups["modules"][name] = item
            else:
                groups["stages"][name] = item
        return {
            "schema": "octo.module_latency.v1",
            "profile_iterations": len(self._iterations),
            "timing_method": "nested CUDA events; one synchronization per inference",
            "notes": [
                "Module and stage CUDA timings are inclusive, so nested values must not be summed.",
                "CUDA-event intervals also include host gaps between the two event records; CPU-only ranges such as attention-mask construction therefore represent elapsed stage latency, not GPU-kernel time.",
                "host_enqueue_per_call is Python enqueue time, not synchronized module latency.",
                "This profile is collected separately and is excluded from policy latency metrics.",
            ],
            **groups,
        }


def _should_profile_module(path: str) -> bool:
    if re.fullmatch(r"octo_transformer\.task_tokenizers\.[^.]+", path):
        return True
    if re.fullmatch(r"octo_transformer\.observation_tokenizers\.[^.]+", path):
        return True
    if re.fullmatch(
        r"octo_transformer\.observation_tokenizers\.[^.]+\."
        r"(encoder_def|token_learner)",
        path,
    ):
        return True
    if re.fullmatch(
        r"octo_transformer\.observation_tokenizers\.[^.]+\.encoder_def\."
        r"(layers\.\d+|embedding|root|stages\.\d+)",
        path,
    ):
        return True
    if path in {
        "octo_transformer",
        "octo_transformer.block_transformer",
        "octo_transformer.block_transformer.transformer",
        "heads.action",
        "heads.action.map_head",
        "heads.action.mean_proj",
        "heads.action.diffusion_model",
        "heads.action.diffusion_model.time_preprocess",
        "heads.action.diffusion_model.cond_encoder",
        "heads.action.diffusion_model.reverse_network",
    }:
        return True
    if re.fullmatch(
        r"octo_transformer\.block_transformer\.transformer\.encoder_blocks\."
        r"\d+(\.(self_attention|mlp_block))?",
        path,
    ):
        return True
    return bool(
        re.fullmatch(
            r"heads\.action\.diffusion_model\.reverse_network\.blocks\.\d+",
            path,
        )
    )


def install_profile_hooks(
    model: OctoModelPt, recorder: CudaProfileRecorder
) -> list[Any]:
    """Attach hooks only to stable architecture-level module boundaries."""

    handles = []
    for path, module in model.module.named_modules():
        if not _should_profile_module(path):
            continue
        recorder.module_metadata[path] = {
            "module_path": path,
            "module_type": type(module).__name__,
        }
        stack = []

        def pre_hook(_module, _inputs, *, profile_name=path, profile_stack=stack):
            profile_stack.append(recorder.begin_module(profile_name))

        def post_hook(
            _module,
            _inputs,
            _output,
            *,
            profile_stack=stack,
        ):
            recorder.end_module(profile_stack.pop())

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return handles


def resize_image(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    return np.asarray(
        Image.fromarray(image).convert("RGB").resize((256, 256)),
        dtype=np.uint8,
    ).copy()


def load_dataset_sample(
    dataset_path: Path,
    split: str,
    episode_index: int,
    step_index: int,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str, Dict[str, Any]]:
    """Load a history window ending at one step from the local RLDS sample."""

    import tensorflow as tf
    import tensorflow_datasets as tfds

    # TensorFlow is only a data reader in this benchmark. Reserve the GPU for
    # PyTorch before either runtime initializes a CUDA context.
    tf.config.set_visible_devices([], "GPU")
    tf.get_logger().setLevel("ERROR")
    builder = tfds.builder_from_directory(str(dataset_path))
    episodes = builder.as_dataset(
        split=split,
        shuffle_files=False,
        read_config=tfds.ReadConfig(try_autocache=False),
    )
    episode = next(iter(episodes.skip(episode_index).take(1)), None)
    if episode is None:
        raise IndexError(f"Episode {episode_index} does not exist in split {split}")

    steps = list(episode["steps"].as_numpy_iterator())
    requested_step_index = step_index
    if step_index < 0:
        step_index += len(steps)
    if not 0 <= step_index < len(steps):
        raise IndexError(
            f"Step {requested_step_index} is outside episode length {len(steps)}. "
            f"Valid indices: 0..{len(steps) - 1} or -{len(steps)}..-1."
        )

    images = []
    timestep_mask = []
    first_history_index = step_index - window_size + 1
    for history_index in range(first_history_index, step_index + 1):
        valid = history_index >= 0
        source_index = max(history_index, 0)
        images.append(resize_image(steps[source_index]["observation"]["image_0"]))
        timestep_mask.append(valid)

    instruction = steps[step_index]["language_instruction"].decode("utf-8")
    metadata = {
        "source": "rlds_sample",
        "dataset_path": str(dataset_path),
        "split": split,
        "episode_index": episode_index,
        "step_index": step_index,
        "requested_step_index": requested_step_index,
        "episode_length": len(steps),
        "language_instruction": instruction,
        "raw_dataset_action": steps[step_index]["action"].tolist(),
    }
    return (
        np.stack(images)[None],
        np.asarray(timestep_mask, dtype=bool)[None],
        None,
        instruction,
        metadata,
    )


def _fetch_json(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "octo-pytorch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _decode_video_frame(video_path: Path, frame_index: int, fps: float) -> np.ndarray:
    """Decode one AV1 frame using imageio-ffmpeg's bundled software decoder."""

    import imageio_ffmpeg

    if not video_path.is_file():
        raise FileNotFoundError(
            f"ALOHA video not found: {video_path}\n"
            "Download it with:\n"
            "  hf download lerobot/aloha_sim_transfer_cube_scripted "
            "--repo-type dataset --local-dir "
            "data/aloha_sim_transfer_cube_scripted"
        )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{frame_index / fps:.9f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    expected_bytes = 480 * 640 * 3
    if len(process.stdout) != expected_bytes:
        raise RuntimeError(
            f"FFmpeg returned {len(process.stdout)} bytes for frame {frame_index}; "
            f"expected {expected_bytes}"
        )
    return np.frombuffer(process.stdout, dtype=np.uint8).reshape(480, 640, 3)


def load_aloha_hf_sample(
    repo_id: str,
    video_path: Path,
    split: str,
    episode_index: int,
    step_index: int,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Load image/state/action from the small LeRobot ALOHA simulation dataset."""

    dataset_dir = video_path.parents[3]
    local_info_path = dataset_dir / "meta" / "info.json"
    if local_info_path.is_file():
        info = json.loads(local_info_path.read_text())
    else:
        info_url = (
            "https://huggingface.co/datasets/" f"{repo_id}/resolve/main/meta/info.json"
        )
        info = _fetch_json(info_url)
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])

    split_path = dataset_dir / "meta" / "splits.json"
    if split_path.is_file():
        split_manifest = json.loads(split_path.read_text())
        if split not in split_manifest:
            available = ", ".join(
                key for key, value in split_manifest.items() if isinstance(value, list)
            )
            raise KeyError(f"Unknown ALOHA split {split!r}; available: {available}")
        split_episodes = split_manifest[split]
    elif split == "train":
        split_episodes = list(range(total_episodes))
    else:
        raise FileNotFoundError(
            f"ALOHA split manifest not found: {split_path}. Only split='train' "
            "is available without it."
        )

    split_episode_index = episode_index
    if not 0 <= split_episode_index < len(split_episodes):
        raise IndexError(
            f"Episode {split_episode_index} is outside split {split!r} length "
            f"{len(split_episodes)}. Valid indices: 0..{len(split_episodes) - 1}."
        )
    global_episode_index = int(split_episodes[split_episode_index])
    if not 0 <= global_episode_index < total_episodes:
        raise ValueError(
            f"Split {split!r} contains invalid global episode "
            f"{global_episode_index}"
        )
    if total_frames % total_episodes:
        raise ValueError("This loader expects equal-length ALOHA simulation episodes")
    episode_length = total_frames // total_episodes
    requested_step_index = step_index
    if step_index < 0:
        step_index += episode_length
    if not 0 <= step_index < episode_length:
        raise IndexError(
            f"Step {requested_step_index} is outside episode length {episode_length}. "
            f"Valid indices: 0..{episode_length - 1} or "
            f"-{episode_length}..-1."
        )

    first_history_step = max(0, step_index - window_size + 1)
    history_length = step_index - first_history_step + 1
    global_offset = global_episode_index * episode_length + first_history_step
    query = urllib.parse.urlencode(
        {
            "dataset": repo_id,
            "config": "default",
            "split": "train",
            "offset": global_offset,
            "length": history_length,
        }
    )
    rows_payload = _fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")
    rows = [item["row"] for item in rows_payload["rows"]]
    if len(rows) != history_length:
        raise RuntimeError(
            f"Dataset server returned {len(rows)} rows; expected {history_length}"
        )

    padding = window_size - history_length
    padded_rows = [rows[0]] * padding + rows
    global_indices = [
        global_episode_index * episode_length + int(row["frame_index"])
        for row in padded_rows
    ]
    fps = float(info["fps"])
    images = [
        resize_image(_decode_video_frame(video_path, index, fps))
        for index in global_indices
    ]
    proprio = np.asarray(
        [row["observation.state"] for row in padded_rows], dtype=np.float32
    )[None]
    timestep_mask = np.asarray([False] * padding + [True] * history_length, dtype=bool)[
        None
    ]
    instruction = "transfer the red cube to the other arm"
    selected_row = rows[-1]
    metadata = {
        "source": "aloha_hf",
        "repo_id": repo_id,
        "video_path": str(video_path),
        "camera": "observation.images.top",
        "fps": fps,
        "split": split,
        "split_episode_index": split_episode_index,
        "global_episode_index": global_episode_index,
        "step_index": step_index,
        "requested_step_index": requested_step_index,
        "global_frame_index": global_episode_index * episode_length + step_index,
        "episode_length": episode_length,
        "language_instruction": instruction,
        "raw_dataset_state": selected_row["observation.state"],
        "raw_dataset_action": selected_row["action"],
    }
    return (
        np.stack(images)[None],
        timestep_mask,
        proprio,
        instruction,
        metadata,
    )


def load_golden_sample(
    golden_dir: Path,
    sample_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Load a deterministic exported golden sample."""

    from PIL import Image

    manifest_path = golden_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Golden manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples", [])
    if not 0 <= sample_index < len(samples):
        raise IndexError(
            f"Golden sample {sample_index} is outside set length {len(samples)}. "
            f"Valid indices: 0..{len(samples) - 1}."
        )

    sample = samples[sample_index]
    sample_dir = Path(sample["path"])
    if not sample_dir.is_absolute() and not sample_dir.exists():
        sample_dir = golden_dir / sample_dir
    if not sample_dir.is_absolute():
        sample_dir = (
            golden_dir.parent / sample_dir.name
            if sample_dir.parent == golden_dir
            else sample_dir
        )
    if not sample_dir.is_dir():
        # Manifest paths are written relative to the invocation cwd in older
        # artifacts. Fall back to sample_id under golden_dir.
        sample_dir = golden_dir / sample["sample_id"]

    window_size = int(manifest["window_size"])
    images = []
    for history_index in range(window_size):
        images.append(
            np.asarray(
                Image.open(sample_dir / f"primary_{history_index}.png").convert("RGB"),
                dtype=np.uint8,
            )
        )
    proprio = np.load(sample_dir / "proprio.npy").astype(np.float32)[None]
    timestep_mask = np.load(sample_dir / "timestep_pad_mask.npy").astype(bool)[None]
    action_label = np.load(sample_dir / "action_label.npy").astype(np.float32)
    instruction = (sample_dir / "instruction.txt").read_text(encoding="utf-8").strip()
    metadata_path = sample_dir / "metadata.json"
    sample_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    metadata = {
        "source": "golden",
        "golden_dir": str(golden_dir),
        "sample_id": sample["sample_id"],
        "sample_index": sample_index,
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("dataset_version"),
        "split": manifest.get("split"),
        "episode_index": sample.get("episode_index"),
        "step_index": sample.get("step_index"),
        "window_size": window_size,
        "action_horizon": int(manifest.get("action_horizon", action_label.shape[0])),
        "language_instruction": instruction,
        "raw_dataset_action": action_label[0].tolist(),
        "raw_dataset_action_chunk": action_label.tolist(),
        "file_checksums": sample.get(
            "file_checksums", sample_metadata.get("file_checksums", {})
        ),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }
    return (
        np.stack(images)[None],
        timestep_mask,
        proprio,
        instruction,
        metadata,
    )


def load_input(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str, Dict[str, Any]]:
    if args.image is None:
        if args.input_source == "golden":
            return load_golden_sample(args.golden_dir, args.episode_index)
        if args.input_source == "aloha-hf":
            return load_aloha_hf_sample(
                args.aloha_repo_id,
                args.aloha_video,
                args.split,
                args.episode_index,
                args.step_index,
                args.window_size,
            )
        return load_dataset_sample(
            args.sample_dataset,
            args.split,
            args.episode_index,
            args.step_index,
            args.window_size,
        )

    from PIL import Image

    image = resize_image(np.asarray(Image.open(args.image).convert("RGB")))
    images = np.repeat(image[None, None], args.window_size, axis=1)
    instruction = args.instruction or "pick up the fork"
    return (
        images,
        np.ones(images.shape[:2], dtype=bool),
        None,
        instruction,
        {"source": "image", "image_path": str(args.image)},
    )


def make_observation(
    host_images: np.ndarray,
    timestep_mask: np.ndarray,
    device: torch.device,
    host_proprio: Optional[np.ndarray] = None,
    include_proprio: bool = False,
    action_horizon: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    image_tensor = torch.from_numpy(host_images.transpose((0, 1, 4, 2, 3)).copy()).to(
        device
    )
    timestep_mask_tensor = torch.from_numpy(timestep_mask).to(device)
    batch_size, window_size = timestep_mask.shape
    observation = {
        "image_primary": image_tensor,
        "timestep": torch.arange(window_size, dtype=torch.int32, device=device)
        .reshape(1, window_size)
        .expand(batch_size, window_size),
        "pad_mask_dict": {
            "image_primary": timestep_mask_tensor,
            "timestep": timestep_mask_tensor,
        },
        "timestep_pad_mask": timestep_mask_tensor,
    }
    if include_proprio:
        if host_proprio is None:
            raise ValueError("The checkpoint requires proprio but the input has none")
        observation["proprio"] = torch.from_numpy(host_proprio).to(device)
        observation["pad_mask_dict"]["proprio"] = timestep_mask_tensor
    if action_horizon is not None:
        observation["task_completed"] = torch.zeros(
            (batch_size, window_size, action_horizon),
            dtype=torch.bool,
            device=device,
        )
    return observation


def resolve_action_statistics(model: OctoModelPt, dataset: str) -> Dict[str, Any]:
    """Handle both base-checkpoint and single-dataset fine-tune statistics."""

    statistics = model.dataset_statistics
    if "action" in statistics:
        return statistics["action"]
    if dataset not in statistics:
        available = ", ".join(sorted(statistics))
        raise KeyError(
            f"Dataset statistics {dataset!r} not found; available: {available}"
        )
    return statistics[dataset]["action"]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    (
        host_images,
        timestep_mask,
        host_proprio,
        sample_instruction,
        input_metadata,
    ) = load_input(args)
    instruction = args.instruction or sample_instruction
    input_metadata["language_instruction"] = instruction
    print(
        f"Input={input_metadata['source']} instruction={instruction!r} "
        f"images={host_images.shape} mask={timestep_mask.tolist()}",
        flush=True,
    )

    fmt = checkpoint_format(args.checkpoint, args.checkpoint_format)
    print(f"Loading {fmt} checkpoint: {args.checkpoint}", flush=True)
    load_start = time.perf_counter()
    resolved_checkpoint_step = resolve_local_checkpoint_step(
        args.checkpoint, args.checkpoint_step
    ) if fmt == "pytorch" else args.checkpoint_step
    model = load_model(args.checkpoint, fmt, resolved_checkpoint_step).to(device).eval()
    load_time_s = time.perf_counter() - load_start

    expected_observations = model.example_batch["observation"]
    include_proprio = "proprio" in expected_observations
    expected_task_completed = expected_observations.get("task_completed")
    action_horizon_for_observation = (
        int(expected_task_completed.shape[-1])
        if expected_task_completed is not None
        else None
    )
    observation = make_observation(
        host_images,
        timestep_mask,
        device,
        host_proprio=host_proprio,
        include_proprio=include_proprio,
        action_horizon=action_horizon_for_observation,
    )
    task = model.create_tasks(texts=[instruction], device=device)
    statistics = resolve_action_statistics(model, args.dataset)
    predicted_action_dim = len(statistics["mean"])
    raw_action = input_metadata.get("raw_dataset_action")
    if raw_action is not None and len(raw_action) != predicted_action_dim:
        print(
            "WARNING: input action has "
            f"{len(raw_action)} dimensions but checkpoint outputs "
            f"{predicted_action_dim}. The run is valid for latency/runtime testing, "
            "but its action is not executable on ALOHA. Use an ALOHA-finetuned "
            "PyTorch checkpoint for 14-D robot actions.",
            flush=True,
        )
    generator = torch.Generator(device=device).manual_seed(0)
    if not args.show_model_warnings:
        # Missing optional wrist/state inputs are expected in this image-only
        # benchmark. Logging them inside every measured call would skew wall time.
        logging.getLogger().setLevel(logging.ERROR)

    def infer(
        prepared_observation: Dict[str, torch.Tensor],
        profiler: Optional[CudaProfileRecorder] = None,
    ) -> torch.Tensor:
        return model.sample_actions(
            prepared_observation,
            task,
            unnormalization_statistics=statistics,
            generator=generator,
            save_attention_mask=False,
            profiler=profiler,
        )

    print(
        f"Backend=pytorch device={torch.cuda.get_device_name(device)} "
        f"torch={torch.__version__} CUDA={torch.version.cuda}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    module_profile = None

    with torch.inference_mode():
        action, cold_wall_ms, cold_gpu_ms = measure_cuda_call(
            lambda: infer(observation), device
        )
        print(
            f"cold_start: wall={cold_wall_ms:.2f} ms gpu={cold_gpu_ms:.2f} ms",
            flush=True,
        )

        for index in range(args.warmup):
            measure_cuda_call(lambda: infer(observation), device)
            print(f"warmup {index + 1}/{args.warmup}", flush=True)

        policy_wall = LatencyTracker("policy_gpu_wall")
        policy_cuda = LatencyTracker("policy_gpu_cuda")
        for index in range(args.runs):
            action, wall_ms, gpu_ms = measure_cuda_call(
                lambda: infer(observation), device
            )
            policy_wall.record(wall_ms)
            policy_cuda.record(gpu_ms)
            print(
                f"policy_gpu iteration={index + 1:03d} "
                f"wall={wall_ms:.2f} ms gpu={gpu_ms:.2f} ms",
                flush=True,
            )

        e2e_wall = LatencyTracker("policy_e2e_wall")
        e2e_cuda = LatencyTracker("policy_e2e_cuda")
        for index in range(args.runs):

            def host_to_action() -> torch.Tensor:
                return infer(
                    make_observation(
                        host_images,
                        timestep_mask,
                        device,
                        host_proprio=host_proprio,
                        include_proprio=include_proprio,
                        action_horizon=action_horizon_for_observation,
                    )
                )

            action, wall_ms, gpu_ms = measure_cuda_call(host_to_action, device)
            e2e_wall.record(wall_ms)
            e2e_cuda.record(gpu_ms)
            print(
                f"policy_e2e iteration={index + 1:03d} "
                f"wall={wall_ms:.2f} ms gpu={gpu_ms:.2f} ms",
                flush=True,
            )

        if args.profile_modules:
            print(
                f"\nModule profile: {args.profile_runs} separate iterations",
                flush=True,
            )
            task_creation_wall = LatencyTracker("task_creation_wall")
            observation_h2d_wall = LatencyTracker("observation_h2d_wall")
            observation_h2d_cuda = LatencyTracker("observation_h2d_cuda")
            for _ in range(args.profile_runs):
                torch.cuda.synchronize(device)
                task_start = time.perf_counter_ns()
                model.create_tasks(texts=[instruction], device=device)
                torch.cuda.synchronize(device)
                task_creation_wall.record(
                    (time.perf_counter_ns() - task_start) / 1_000_000.0
                )
                _, h2d_wall_ms, h2d_cuda_ms = measure_cuda_call(
                    lambda: make_observation(
                        host_images,
                        timestep_mask,
                        device,
                        host_proprio=host_proprio,
                        include_proprio=include_proprio,
                        action_horizon=action_horizon_for_observation,
                    ),
                    device,
                )
                observation_h2d_wall.record(h2d_wall_ms)
                observation_h2d_cuda.record(h2d_cuda_ms)

            recorder = CudaProfileRecorder(device)
            hook_handles = install_profile_hooks(model, recorder)
            try:
                for index in range(args.profile_runs):
                    recorder.begin_iteration()
                    with recorder.range("profiled_policy_total"):
                        action = infer(observation, profiler=recorder)
                    recorder.end_iteration()
                    print(
                        f"module_profile {index + 1}/{args.profile_runs}",
                        flush=True,
                    )
            finally:
                for handle in hook_handles:
                    handle.remove()

            module_profile = recorder.report()
            module_profile["preprocessing"] = {
                "task_creation_wall": task_creation_wall.summary(),
                "observation_h2d_wall": observation_h2d_wall.summary(),
                "observation_h2d_cuda": observation_h2d_cuda.summary(),
            }
            print("Module CUDA summary (inclusive)", flush=True)
            for name, profile in module_profile["stages"].items():
                mean_ms = profile["cuda_per_inference"]["mean_ms"]
                calls = profile["calls_per_inference"]["mean"]
                print(
                    f"  stage {name}: {mean_ms:.2f} ms/inference " f"calls={calls:.1f}",
                    flush=True,
                )
            for name, profile in module_profile["modules"].items():
                mean_ms = profile["cuda_per_inference"]["mean_ms"]
                calls = profile["calls_per_inference"]["mean"]
                print(
                    f"  module {name}: {mean_ms:.2f} ms/inference "
                    f"calls={calls:.1f}",
                    flush=True,
                )

    metrics = {
        tracker.name: tracker.summary()
        for tracker in (policy_wall, policy_cuda, e2e_wall, e2e_cuda)
    }
    print("\nLatency summary (warm runs)")
    print(format_latency_summary(metrics["policy_gpu_wall"]))
    print(format_latency_summary(metrics["policy_e2e_wall"]))

    result = {
        "schema": "octo.policy_latency.v1",
        "pipeline_fingerprint": args.pipeline_fingerprint,
        "backend": "pytorch",
        "checkpoint": args.checkpoint,
        "checkpoint_step": resolved_checkpoint_step,
        "checkpoint_format": fmt,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "batch_size": 1,
        "window_size": args.window_size,
        "model_uses_proprio": include_proprio,
        "input": input_metadata,
        "warmup_iterations": args.warmup,
        "benchmark_iterations": args.runs,
        "module_profile_iterations": args.profile_runs if args.profile_modules else 0,
        "model_load_s": load_time_s,
        "cold_start_wall_ms": cold_wall_ms,
        "cold_start_cuda_ms": cold_gpu_ms,
        "action_shape": list(action.shape),
        "sample_action": action.tolist(),
        "input_checksum_sha256": hashlib.sha256(
            json.dumps(input_metadata, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "output_action_checksum_sha256": hashlib.sha256(
            np.asarray(action.detach().cpu(), dtype=np.float32).tobytes()
        ).hexdigest(),
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "scope_definitions": {
            "policy_gpu": (
                "GPU-resident observation and cached task tokens to CPU action"
            ),
            "policy_e2e": (
                "256x256 NumPy observation on CPU through H2D transfer to CPU "
                "action; task tokens cached"
            ),
        },
        "metrics": metrics,
        "module_profile": module_profile,
    }

    print(
        f"Action shape={tuple(action.shape)} "
        f"peak_allocated={result['peak_memory_allocated_mib']:.1f} MiB "
        f"peak_reserved={result['peak_memory_reserved_mib']:.1f} MiB",
        flush=True,
    )
    print(
        "Sample action:\n"
        + np.array2string(action.numpy(), precision=4, suppress_small=True),
        flush=True,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved JSON: {args.output_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
