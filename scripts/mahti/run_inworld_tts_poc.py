from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from common import (
    DEFAULT_MANIFEST,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "tts_naturalness"
DEFAULT_CHUNK_TEXT_DIR = (
    REPO_ROOT
    / "generated"
    / "audiobooks-half_of_a_yellow_sun-full"
    / "half_of_a_yellow_sun"
    / "chunks"
)
DEFAULT_CHUNK_AUDIO_DIR = (
    REPO_ROOT
    / "generated"
    / "audiobooks-half_of_a_yellow_sun-full"
    / "half_of_a_yellow_sun"
    / "qwen3_faster_clone"
    / "chunks"
)
DEFAULT_PROMPT_WAV = (
    REPO_ROOT / "assets" / "audio_samples" / "brothersKarmatzov.clone_ref_15s.wav"
)
DEFAULT_PROMPT_TRANSCRIPT = DEFAULT_PROMPT_WAV.with_suffix(".transcript.txt")
DEFAULT_TEXT_FILE = EXPERIMENT_ROOT / "inputs" / "long_audiobook_test.txt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "generated" / "inworld_tts_training"
DEFAULT_BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_VOICE_DESCRIPTION = (
    "Warm, steady audiobook narrator with clear diction, restrained emotion, and a "
    "mature male voice."
)
DEFAULT_SEGMENT_TARGET_SECONDS = 24.0
DEFAULT_SEGMENT_MAX_SECONDS = 28.0
DEFAULT_SEGMENT_MIN_SECONDS = 8.0
SEGMENT_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


@dataclasses.dataclass(frozen=True)
class SynthesisPreset:
    id: str
    mode: str
    voice_description: str
    enable_instruction: bool
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    frequency_penalty: float
    max_tokens: int
    min_tokens: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a local Inworld proof-of-concept model on the existing Qwen clone "
            "chunk corpus, then render a small clone/non-clone sample matrix."
        )
    )
    parser.add_argument("--chunk-text-dir", type=Path, default=DEFAULT_CHUNK_TEXT_DIR)
    parser.add_argument("--chunk-audio-dir", type=Path, default=DEFAULT_CHUNK_AUDIO_DIR)
    parser.add_argument("--prompt-wav", type=Path, default=DEFAULT_PROMPT_WAV)
    parser.add_argument(
        "--prompt-transcript-file",
        type=Path,
        default=DEFAULT_PROMPT_TRANSCRIPT,
    )
    parser.add_argument("--text-file", type=Path, default=DEFAULT_TEXT_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--serving-dir",
        type=Path,
        default=None,
        help="Optional directory for the converted serving export. Defaults to <run>/serving.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to resume training from or convert directly.",
    )
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume training from the selected checkpoint instead of skipping straight to inference.",
    )
    parser.add_argument(
        "--resume-model-weights-only",
        action="store_true",
        help="When resuming training, load checkpoint model weights but reset optimizer and scheduler state.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--voice-description", default=DEFAULT_VOICE_DESCRIPTION)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum paired chunk samples to use. 0 means all available samples.",
    )
    parser.add_argument(
        "--train-epochs",
        type=float,
        default=32.0,
        help="Effective dataset epochs for training via weighted datasets.",
    )
    parser.add_argument(
        "--val-epochs",
        type=float,
        default=1.0,
        help="Effective dataset epochs for validation via weighted datasets.",
    )
    parser.add_argument(
        "--target-train-epochs",
        type=float,
        default=None,
        help="Optional effective train epochs override when resuming from an existing training config.",
    )
    parser.add_argument(
        "--target-val-epochs",
        type=float,
        default=None,
        help="Optional effective validation epochs override when resuming from an existing training config.",
    )
    parser.add_argument(
        "--target-learning-rate",
        type=float,
        default=None,
        help="Optional learning-rate override when resuming from an existing training config.",
    )
    parser.add_argument("--vectorizer-batch-size", type=int, default=8)
    parser.add_argument("--vectorizer-num-workers", type=int, default=2)
    parser.add_argument(
        "--segment-target-seconds",
        type=float,
        default=DEFAULT_SEGMENT_TARGET_SECONDS,
        help="Target duration for each derived training utterance.",
    )
    parser.add_argument(
        "--segment-max-seconds",
        type=float,
        default=DEFAULT_SEGMENT_MAX_SECONDS,
        help="Hard cap for each derived training utterance to satisfy Inworld filters.",
    )
    parser.add_argument(
        "--segment-min-seconds",
        type=float,
        default=DEFAULT_SEGMENT_MIN_SECONDS,
        help="Minimum desired duration before merging a tiny trailing segment back.",
    )
    parser.add_argument("--training-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--training-strategy",
        choices=("ddp", "fsdp", "deepspeed"),
        default="ddp",
    )
    parser.add_argument(
        "--distributed-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use for vectorization and training inside the allocation.",
    )
    parser.add_argument(
        "--cpus-per-gpu",
        type=int,
        default=8,
        help="CPU threads reserved per distributed GPU task.",
    )
    parser.add_argument(
        "--preset-ids",
        nargs="*",
        default=None,
        help="Optional subset of synthesis preset ids to render.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Optional checkpoint save frequency override.",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=None,
        help="Optional checkpoint retention override.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def run_command(command: list[str], *, cwd: Path) -> None:
    print(f"[inworld-poc] running: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def checkpoint_size_bytes(checkpoint_path: Path) -> int:
    try:
        return checkpoint_path.resolve(strict=True).stat().st_size
    except (FileNotFoundError, OSError):
        return 0


def select_recoverable_checkpoint(training_dir: Path) -> Path | None:
    candidates: list[tuple[int, int, Path]] = []
    for checkpoint_path in training_dir.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", checkpoint_path.name)
        if not match:
            continue
        size_bytes = checkpoint_size_bytes(checkpoint_path)
        if size_bytes <= 0:
            continue
        candidates.append((int(match.group(1)), size_bytes, checkpoint_path))

    if not candidates:
        final_checkpoint = training_dir / "final_model.pt"
        return final_checkpoint if checkpoint_size_bytes(final_checkpoint) > 0 else None

    max_size = max(size_bytes for _, size_bytes, _ in candidates)
    min_recoverable_size = max(1, int(max_size * 0.9))

    final_checkpoint = training_dir / "final_model.pt"
    final_size = checkpoint_size_bytes(final_checkpoint)
    if final_size >= min_recoverable_size:
        return final_checkpoint

    recoverable = [
        candidate for candidate in candidates if candidate[1] >= min_recoverable_size
    ]
    if not recoverable:
        recoverable = [max(candidates, key=lambda item: item[1])]

    _, _, checkpoint_path = max(recoverable, key=lambda item: item[0])
    return checkpoint_path


def serving_export_is_valid(serving_dir: Path) -> bool:
    required_files = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if not serving_dir.exists():
        return False
    if any(not (serving_dir / name).is_file() for name in required_files):
        return False

    model_files = sorted(serving_dir.glob("*.safetensors"))
    if not model_files:
        return False

    total_model_bytes = 0
    try:
        from safetensors import safe_open
    except Exception:
        safe_open = None

    for model_file in model_files:
        try:
            size_bytes = model_file.stat().st_size
        except FileNotFoundError:
            return False
        if size_bytes < 1024 * 1024:
            return False
        total_model_bytes += size_bytes
        if safe_open is not None:
            try:
                with safe_open(str(model_file), framework="pt") as handle:
                    tuple(handle.keys())
            except Exception:
                return False

    return total_model_bytes >= 512 * 1024 * 1024


def convert_checkpoint_to_serving(
    *,
    inworld_root: Path,
    selected_checkpoint: Path,
    serving_dir: Path,
) -> None:
    tmp_serving_dir = serving_dir.parent / f".{serving_dir.name}.tmp-{os.getpid()}"

    if tmp_serving_dir.exists():
        shutil.rmtree(tmp_serving_dir, ignore_errors=True)

    try:
        run_command(
            [
                "python",
                "tools/serving/convert_checkpoint.py",
                "--checkpoint_path",
                str(selected_checkpoint),
                "--output_path",
                str(tmp_serving_dir),
            ],
            cwd=inworld_root,
        )
        if not serving_export_is_valid(tmp_serving_dir):
            raise RuntimeError(
                f"converted serving export is incomplete or unreadable: {tmp_serving_dir}"
            )
        if serving_dir.exists():
            shutil.rmtree(serving_dir)
        tmp_serving_dir.replace(serving_dir)
    except Exception:
        if tmp_serving_dir.exists():
            shutil.rmtree(tmp_serving_dir, ignore_errors=True)
        raise


def require_existing_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(":", "_")
        .replace(".", "p")
        .replace("-", "_")
    )


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name

    sample_count = "all" if args.max_samples <= 0 else str(args.max_samples)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return (
        f"inworld_{safe_name(args.base_model)}_s{sample_count}"
        f"_e{safe_name(str(args.train_epochs))}_{timestamp}"
    )


def count_words(text: str) -> int:
    return len([token for token in text.split() if token])


def split_oversized_unit(unit: str, *, target_words: int, max_words: int) -> list[str]:
    words = unit.split()
    if len(words) <= max_words:
        return [unit]

    chunk_size = max(1, min(max_words, target_words))
    return [
        " ".join(words[index : index + chunk_size]).strip()
        for index in range(0, len(words), chunk_size)
        if words[index : index + chunk_size]
    ]


def build_segment_texts(
    *,
    transcript: str,
    total_duration: float,
    target_seconds: float,
    max_seconds: float,
    min_seconds: float,
) -> list[str]:
    total_words = count_words(transcript)
    if total_words <= 0:
        return []

    seconds_per_word = max(total_duration / total_words, 1e-3)
    target_words = max(24, int(round(target_seconds / seconds_per_word)))
    max_words = max(target_words, int(round(max_seconds / seconds_per_word)))
    min_words = max(8, int(round(min_seconds / seconds_per_word)))

    units: list[str] = []
    for raw_unit in SEGMENT_BOUNDARY_RE.split(transcript):
        unit = raw_unit.strip()
        if not unit:
            continue
        units.extend(
            split_oversized_unit(unit, target_words=target_words, max_words=max_words)
        )
    if not units:
        units = [transcript]

    segments: list[str] = []
    current_units: list[str] = []
    current_words = 0
    for unit in units:
        unit_words = count_words(unit)
        if current_units and current_words + unit_words > max_words:
            segments.append(" ".join(current_units).strip())
            current_units = [unit]
            current_words = unit_words
        else:
            current_units.append(unit)
            current_words += unit_words

        if current_words >= target_words:
            segments.append(" ".join(current_units).strip())
            current_units = []
            current_words = 0

    if current_units:
        segments.append(" ".join(current_units).strip())

    if len(segments) > 1:
        trailing_words = count_words(segments[-1])
        previous_words = count_words(segments[-2])
        merge_budget = max_words + max(16, max_words // 8)
        if trailing_words < min_words and previous_words + trailing_words <= merge_budget:
            segments[-2] = f"{segments[-2]} {segments[-1]}".strip()
            segments.pop()

    return [segment for segment in segments if segment]


def build_dataset_jsonl(
    *,
    chunk_text_dir: Path,
    chunk_audio_dir: Path,
    output_jsonl: Path,
    max_samples: int,
    target_seconds: float,
    max_seconds: float,
    min_seconds: float,
) -> tuple[list[dict[str, object]], float, int]:
    wav_paths = sorted(chunk_audio_dir.glob("chunk_*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"no chunk WAV files found in {chunk_audio_dir}")

    derived_audio_dir = output_jsonl.parent / "audio_segments"
    if derived_audio_dir.exists():
        shutil.rmtree(derived_audio_dir)
    derived_audio_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    total_duration = 0.0
    source_pairs = 0
    for wav_path in wav_paths:
        if max_samples > 0 and source_pairs >= max_samples:
            break

        text_path = chunk_text_dir / f"{wav_path.stem}.txt"
        if not text_path.exists():
            continue

        transcript = normalize_target_text(load_text(text_path))
        info = sf.info(str(wav_path))
        duration = float(info.duration)
        segment_texts = build_segment_texts(
            transcript=transcript,
            total_duration=duration,
            target_seconds=target_seconds,
            max_seconds=max_seconds,
            min_seconds=min_seconds,
        )
        if not segment_texts:
            continue

        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        total_segment_words = sum(count_words(segment) for segment in segment_texts)
        if total_segment_words <= 0:
            continue

        consumed_words = 0
        total_frames = audio.shape[0]
        for segment_index, segment_text in enumerate(segment_texts):
            segment_words = count_words(segment_text)
            if segment_words <= 0:
                continue

            start_ratio = consumed_words / total_segment_words
            consumed_words += segment_words
            end_ratio = 1.0 if segment_index == len(segment_texts) - 1 else consumed_words / total_segment_words
            start_frame = max(0, min(total_frames - 1, int(round(start_ratio * total_frames))))
            end_frame = max(start_frame + 1, min(total_frames, int(round(end_ratio * total_frames))))
            segment_audio = audio[start_frame:end_frame]
            segment_duration = float(segment_audio.shape[0]) / float(sample_rate)

            segment_id = f"{wav_path.stem}_seg{segment_index:02d}"
            segment_wav_path = derived_audio_dir / f"{segment_id}.wav"
            sf.write(str(segment_wav_path), segment_audio, sample_rate)

            total_duration += segment_duration
            entries.append(
                {
                    "id": segment_id,
                    "wav_path": str(segment_wav_path.resolve()),
                    "speaker_id": "qwen_clone_corpus",
                    "language": "en",
                    "transcript": segment_text,
                    "duration": round(segment_duration, 4),
                    "sample_rate": int(sample_rate),
                    "source_id": wav_path.stem,
                }
            )

        source_pairs += 1

    if not entries:
        raise RuntimeError("no paired text/audio samples were found for the Inworld proof corpus")

    ensure_parent(output_jsonl)
    output_jsonl.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return entries, round(total_duration, 4), source_pairs


def normalize_single_rank_vectorized_dir(vectorized_dir: Path) -> None:
    for split in ("train", "val"):
        shard_specs = (
            ("codes", "npy"),
            ("codes_index", "npy"),
            ("samples", "jsonl"),
        )
        for stem, extension in shard_specs:
            pattern = re.compile(rf"^{split}_{stem}_-?\d+\.{extension}$")
            shard_matches = sorted(
                path
                for path in vectorized_dir.iterdir()
                if path.is_file() and pattern.match(path.name)
            )
            if len(shard_matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {split} shard for {stem}.{extension} in {vectorized_dir}, "
                    f"found {len(shard_matches)}"
                )

            destination = vectorized_dir / f"{split}_{stem}.{extension}"
            if destination.exists():
                destination.unlink()
            shard_matches[0].replace(destination)


def write_sft_config(
    config_path: Path,
    *,
    vectorized_dir: Path,
    model_name: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    strategy: str,
    train_epochs: float,
    val_epochs: float,
) -> dict[str, object]:
    config = {
        "training": {
            "seed": 777,
            "logging_steps": 10,
            "eval_steps": 50,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clip_value": 1.0,
            "learning_rate": learning_rate,
            "betas": [0.9, 0.95],
            "warmup_ratio": 0.05,
            "batch_size": batch_size,
            "weight_decay": 0.1,
            "precision": "bf16",
            "strategy": strategy,
            "gradient_checkpointing": True,
            "num_workers": 0,
        },
        "modeling": {
            "parameters": {
                "codebook_size": 65536,
                "max_seq_len": 2048,
                "model_name": model_name,
                "enable_text_normalization": True,
            }
        },
        "checkpointing": {
            "save_steps": 100,
            "collect_health_stats": False,
            "save_intermediate_generations": False,
            "only_load_model_weights": True,
            "keep_only_last_n_checkpoints": 1,
        },
        "train_weighted_datasets": {
            str(vectorized_dir.resolve()): train_epochs,
        },
        "val_weighted_datasets": {
            str(vectorized_dir.resolve()): val_epochs,
        },
        "dataset": {
            "allowed_languages": ["en"],
            "min_dnsmos_score": 0,
            "min_sample_rate": 24000,
            "enable_rlhf_training": False,
            "min_audio_duration": 0.0,
        },
    }
    ensure_parent(config_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def load_existing_training_config(
    *, training_dir: Path, config_path: Path, resume_checkpoint: Path | None = None
) -> dict[str, object] | None:
    candidate_paths = [
        training_dir / "training_config.json",
        config_path,
    ]
    if resume_checkpoint is not None:
        candidate_paths.insert(0, resume_checkpoint.parent / "training_config.json")
    for candidate_path in candidate_paths:
        if not candidate_path.is_file():
            continue
        try:
            return json.loads(candidate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return None


def normalize_training_config(
    *,
    config: dict[str, object],
    training_dir: Path,
    vectorized_dir: Path,
    args: argparse.Namespace,
    resume_checkpoint: Path | None,
) -> dict[str, object]:
    training = dict(config.get("training", {}))
    modeling = dict(config.get("modeling", {}))
    checkpointing = dict(config.get("checkpointing", {}))
    dataset = dict(config.get("dataset", {}))
    existing_train_weighted_datasets = config.get("train_weighted_datasets")
    existing_val_weighted_datasets = config.get("val_weighted_datasets")

    if not config.get("training"):
        training = {
            "seed": 777,
            "logging_steps": 10,
            "eval_steps": 50,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_clip_value": 1.0,
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "warmup_ratio": 0.05,
            "batch_size": args.training_batch_size,
            "weight_decay": 0.1,
            "precision": "bf16",
            "strategy": args.training_strategy,
            "gradient_checkpointing": True,
            "num_workers": 0,
        }
    if not config.get("modeling"):
        modeling = {
            "parameters": {
                "codebook_size": 65536,
                "max_seq_len": 2048,
                "model_name": args.base_model,
                "enable_text_normalization": True,
            }
        }
    if not config.get("checkpointing"):
        checkpointing = {
            "save_steps": 100,
            "collect_health_stats": False,
            "save_intermediate_generations": False,
            "only_load_model_weights": True,
            "keep_only_last_n_checkpoints": 1,
        }
    if not config.get("dataset"):
        dataset = {
            "allowed_languages": ["en"],
            "min_dnsmos_score": 0,
            "min_sample_rate": 24000,
            "enable_rlhf_training": False,
            "min_audio_duration": 0.0,
        }

    checkpointing["directory"] = str(training_dir)
    checkpointing.setdefault("save_steps", 100)
    checkpointing.setdefault("collect_health_stats", False)
    checkpointing.setdefault("save_intermediate_generations", False)
    checkpointing.setdefault("only_load_model_weights", True)
    checkpointing.setdefault("keep_only_last_n_checkpoints", 1)

    if args.save_steps is not None:
        checkpointing["save_steps"] = args.save_steps
    if args.keep_last_checkpoints is not None:
        checkpointing["keep_only_last_n_checkpoints"] = args.keep_last_checkpoints

    if args.target_learning_rate is not None:
        training["learning_rate"] = args.target_learning_rate

    if resume_checkpoint is not None:
        checkpointing["checkpoint_file_to_resume_from"] = str(resume_checkpoint)
        checkpointing["only_load_model_weights"] = bool(args.resume_model_weights_only)

    config["training"] = training
    config["modeling"] = modeling
    config["checkpointing"] = checkpointing
    config["dataset"] = dataset
    if args.target_train_epochs is not None or not existing_train_weighted_datasets:
        config["train_weighted_datasets"] = {
            str(vectorized_dir.resolve()): (
                args.target_train_epochs
                if args.target_train_epochs is not None
                else args.train_epochs
            )
        }
    else:
        config["train_weighted_datasets"] = existing_train_weighted_datasets

    if args.target_val_epochs is not None or not existing_val_weighted_datasets:
        config["val_weighted_datasets"] = {
            str(vectorized_dir.resolve()): (
                args.target_val_epochs
                if args.target_val_epochs is not None
                else args.val_epochs
            )
        }
    else:
        config["val_weighted_datasets"] = existing_val_weighted_datasets
    return config


def first_weighted_dataset_value(
    config: dict[str, object], key: str, fallback: float
) -> float:
    weighted_datasets = config.get(key)
    if not isinstance(weighted_datasets, dict) or not weighted_datasets:
        return fallback

    first_value = next(iter(weighted_datasets.values()))
    try:
        return float(first_value)
    except (TypeError, ValueError):
        return fallback


def build_distributed_command(
    *,
    base_command: list[str],
    distributed_gpus: int,
    cpus_per_gpu: int,
) -> list[str]:
    if distributed_gpus <= 1:
        return base_command

    return [
        "srun",
        "--nodes=1",
        f"--ntasks={distributed_gpus}",
        f"--ntasks-per-node={distributed_gpus}",
        "--gpus-per-task=1",
        f"--cpus-per-task={cpus_per_gpu}",
        "--cpu-bind=cores",
        "--kill-on-bad-exit=1",
        *base_command,
    ]


def build_presets(voice_description: str) -> list[SynthesisPreset]:
    return [
        SynthesisPreset(
            id="clone_anchor",
            mode="clone",
            voice_description="",
            enable_instruction=True,
            temperature=0.8,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.4,
            frequency_penalty=0.4,
            max_tokens=3072,
            min_tokens=10,
            seed=42,
        ),
        SynthesisPreset(
            id="clone_looser",
            mode="clone",
            voice_description="",
            enable_instruction=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            frequency_penalty=0.2,
            max_tokens=3072,
            min_tokens=10,
            seed=314,
        ),
        SynthesisPreset(
            id="description_anchor",
            mode="description",
            voice_description=voice_description,
            enable_instruction=False,
            temperature=0.8,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.4,
            frequency_penalty=0.4,
            max_tokens=3072,
            min_tokens=10,
            seed=42,
        ),
        SynthesisPreset(
            id="description_storyteller",
            mode="description",
            voice_description=(
                "Intimate literary storyteller with a low-pitched male timbre, "
                "measured pacing, soft warmth, crisp consonants, and controlled "
                "emphasis suited to long-form narration."
            ),
            enable_instruction=False,
            temperature=0.85,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.3,
            frequency_penalty=0.3,
            max_tokens=3072,
            min_tokens=10,
            seed=2718,
        ),
    ]


def select_presets(
    presets: list[SynthesisPreset], selected_ids: list[str] | None
) -> list[SynthesisPreset]:
    if not selected_ids:
        return presets

    selected = [preset for preset in presets if preset.id in set(selected_ids)]
    missing = sorted(set(selected_ids) - {preset.id for preset in selected})
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Unknown preset ids: {missing_text}")
    return selected


def load_inworld_runtime(
    *,
    inworld_root: Path,
    serving_dir: Path,
    xcodec_checkpoint: Path,
    prompt_wav_path: Path,
) -> dict[str, Any]:
    sys.path.insert(0, str(inworld_root))

    import torch
    import transformers
    from tts.core import constants, prompting
    from tts.core.codec import decoding, encoding
    from tts.data import data_utils
    from tts.inference import inferencing

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(serving_dir))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(serving_dir),
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    audio_encoder = encoding.CachingAudioEncoder(model_path=str(xcodec_checkpoint), device=device)
    audio_decoder = decoding.create(model_path=str(xcodec_checkpoint), device=device)
    prompt_compiler = prompting.InferencePromptCompiler()
    tts_model = inferencing.LocalTtsModel(
        model=model,
        device=device,
        tokenizer=tokenizer,
        audio_encoder=audio_encoder,
        audio_decoder=audio_decoder,
        prompt_compiler=prompt_compiler,
        use_vllm=False,
    )

    prompt_wav, _ = data_utils.load_wav(
        str(prompt_wav_path), target_sample_rate=constants.CODEC_SAMPLE_RATE
    )
    return {
        "device": device,
        "audio_decoder": audio_decoder,
        "tts_model": tts_model,
        "prompt_wav": prompt_wav,
        "inferencing_module": inferencing,
    }


def synthesize_with_runtime(
    *,
    runtime: dict[str, Any],
    preset: SynthesisPreset,
    text_to_synthesize: str,
    prompt_transcript: str,
    output_path: Path,
) -> dict[str, float]:
    inferencing = runtime["inferencing_module"]
    settings = inferencing.InferenceSettings(
        temperature=preset.temperature,
        max_tokens=preset.max_tokens,
        min_tokens=preset.min_tokens,
        top_p=preset.top_p,
        top_k=preset.top_k,
        repetition_penalty=preset.repetition_penalty,
        frequency_penalty=preset.frequency_penalty,
        seed=preset.seed,
    )

    active_prompt_transcript = prompt_transcript if preset.mode == "clone" else ""
    result = runtime["tts_model"].synthesize_speech(
        inference_settings=settings,
        text_to_synthesize=text_to_synthesize,
        prompt_id=output_path.stem,
        prompt_wav=runtime["prompt_wav"],
        audio_prompt_transcription=active_prompt_transcript,
        voice_description=preset.voice_description,
        enable_instruction=preset.enable_instruction,
    )

    ensure_parent(output_path)
    wav_tensor = result.wav.detach().cpu()
    wav_array = wav_tensor.squeeze().numpy()
    sf.write(str(output_path), wav_array, runtime["audio_decoder"].sample_rate)
    info = sf.info(str(output_path))
    duration_sec = float(info.duration)

    return {
        "encoding_time_sec": round(result.encoding_time, 4),
        "decoding_time_sec": round(result.decoding_time, 4),
        "inference_time_sec": round(result.inference_time, 4),
        "duration_sec": round(duration_sec, 4),
        "sample_rate": int(info.samplerate),
    }


def main() -> None:
    args = parse_args()
    run_name = resolve_run_name(args)
    run_root = args.output_root / run_name

    prompt_wav = require_existing_file(args.prompt_wav, "prompt wav")
    prompt_transcript_path = require_existing_file(
        args.prompt_transcript_file, "prompt transcript"
    )
    text_file = require_existing_file(args.text_file, "synthesis text file")
    require_existing_file(args.chunk_text_dir, "chunk text directory")
    require_existing_file(args.chunk_audio_dir, "chunk audio directory")

    text_to_synthesize = normalize_target_text(load_text(text_file))
    prompt_transcript = normalize_target_text(load_text(prompt_transcript_path))

    inworld_root = (
        Path(os.environ.get("MAHTI_EXTERNAL_ROOT", REPO_ROOT / "external")) / "inworld-tts"
    )
    xcodec_checkpoint = (
        Path(os.environ.get("MAHTI_MODEL_ROOT", REPO_ROOT / "models"))
        / "inworld-tts"
        / "xcodec2"
        / "ckpt"
        / "epoch=4-step=1400000.ckpt"
    )

    require_existing_file(inworld_root, "Inworld repo checkout")
    require_existing_file(xcodec_checkpoint, "xcodec2 checkpoint")

    corpus_dir = run_root / "corpus"
    dataset_jsonl = corpus_dir / "samples.jsonl"
    vectorized_dir = run_root / "vectorized"
    config_path = run_root / "configs" / "sft.json"
    training_run_name = "train_main"
    training_experiments_dir = run_root / "training"
    training_dir = training_experiments_dir / training_run_name
    serving_dir = args.serving_dir if args.serving_dir else (run_root / "serving")
    summary_path = run_root / "summary.json"

    if args.force or not dataset_jsonl.exists():
        corpus_entries, corpus_duration_sec, source_pairs = build_dataset_jsonl(
            chunk_text_dir=args.chunk_text_dir,
            chunk_audio_dir=args.chunk_audio_dir,
            output_jsonl=dataset_jsonl,
            max_samples=args.max_samples,
            target_seconds=args.segment_target_seconds,
            max_seconds=args.segment_max_seconds,
            min_seconds=args.segment_min_seconds,
        )
    else:
        corpus_entries = [
            json.loads(line)
            for line in dataset_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        corpus_duration_sec = round(
            sum(float(entry.get("duration", 0.0)) for entry in corpus_entries), 4
        )
        source_pairs = len(
            {
                str(entry.get("source_id", str(entry.get("id", ""))).split("_seg", 1)[0])
                for entry in corpus_entries
            }
        )

    if args.force or not (vectorized_dir / "train_codes.npy").exists():
        vectorizer_command = [
            "python",
            "tools/data/data_vectorizer.py",
            "--codec_model_path",
            str(xcodec_checkpoint),
            "--batch_size",
            str(args.vectorizer_batch_size),
            "--dataset_path",
            str(dataset_jsonl),
            "--output_dir",
            str(vectorized_dir),
            "--num_workers",
            str(args.vectorizer_num_workers),
            "--val_split",
            "0.1",
        ]
        if args.distributed_gpus > 1:
            vectorizer_command.append("--slurm_distributed")
        run_command(
            build_distributed_command(
                base_command=vectorizer_command,
                distributed_gpus=args.distributed_gpus,
                cpus_per_gpu=args.cpus_per_gpu,
            ),
            cwd=inworld_root,
        )
        if args.distributed_gpus > 1:
            run_command(
                [
                    "python",
                    "tools/data/data_merger.py",
                    "--dataset_path",
                    str(vectorized_dir),
                    "--remove_shards",
                ],
                cwd=inworld_root,
            )
        else:
            normalize_single_rank_vectorized_dir(vectorized_dir)

    selected_checkpoint = (
        args.resume_checkpoint if args.resume_checkpoint else select_recoverable_checkpoint(training_dir)
    )
    config = None
    if not args.force and selected_checkpoint is not None:
        config = load_existing_training_config(
            training_dir=training_dir,
            config_path=config_path,
            resume_checkpoint=selected_checkpoint,
        )

    if config is None:
        config = write_sft_config(
            config_path,
            vectorized_dir=vectorized_dir,
            model_name=args.base_model,
            batch_size=args.training_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            strategy=args.training_strategy,
            train_epochs=args.train_epochs,
            val_epochs=args.val_epochs,
        )
    config = normalize_training_config(
        config=config,
        training_dir=training_dir,
        vectorized_dir=vectorized_dir,
        args=args,
        resume_checkpoint=selected_checkpoint if args.resume_training else None,
    )
    effective_train_epochs = first_weighted_dataset_value(
        config,
        "train_weighted_datasets",
        args.train_epochs,
    )
    effective_val_epochs = first_weighted_dataset_value(
        config,
        "val_weighted_datasets",
        args.val_epochs,
    )
    ensure_parent(config_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if args.force or selected_checkpoint is None or args.resume_training:
        if args.distributed_gpus > 1:
            # Fabric expects to launch local worker processes itself for one-node
            # multi-GPU training. Wrapping it in multi-task srun hides all but one
            # GPU from each process and breaks device discovery.
            training_command = [
                "fabric",
                "run",
                "--accelerator=gpu",
                f"--devices={args.distributed_gpus}",
                "--num_nodes=1",
                "tts/training/main.py",
                "--config_path",
                str(config_path),
                "--experiment_dir",
                str(training_experiments_dir),
                "--run_name",
                training_run_name,
            ]
        else:
            training_command = [
                "python",
                "tts/training/main.py",
                "--config_path",
                str(config_path),
                "--experiment_dir",
                str(training_experiments_dir),
                "--run_name",
                training_run_name,
            ]
        try:
            run_command(training_command, cwd=inworld_root)
        except subprocess.CalledProcessError:
            selected_checkpoint = select_recoverable_checkpoint(training_dir)
            if selected_checkpoint is None:
                raise
            print(
                "[inworld-poc] training failed; continuing with recoverable checkpoint "
                f"{selected_checkpoint}",
                flush=True,
            )
        else:
            selected_checkpoint = select_recoverable_checkpoint(training_dir)

    if selected_checkpoint is None:
        raise FileNotFoundError(
            f"no recoverable checkpoint found in training directory: {training_dir}"
        )

    if args.force or not serving_export_is_valid(serving_dir):
        if serving_dir.exists():
            print(
                "[inworld-poc] existing serving export is incomplete or unreadable; "
                f"rebuilding {serving_dir}",
                flush=True,
            )
        convert_checkpoint_to_serving(
            inworld_root=inworld_root,
            selected_checkpoint=selected_checkpoint,
            serving_dir=serving_dir,
        )

    outputs_dir = run_root / "outputs"
    selected_presets = select_presets(build_presets(args.voice_description), args.preset_ids)
    runtime = load_inworld_runtime(
        inworld_root=inworld_root,
        serving_dir=serving_dir,
        xcodec_checkpoint=xcodec_checkpoint,
        prompt_wav_path=prompt_wav,
    )

    summary: dict[str, object] = {
        "cluster": resolve_cluster_name(),
        "run_name": run_name,
        "inworld_root": str(inworld_root),
        "base_model": args.base_model,
        "training_samples": len(corpus_entries),
        "source_pairs": source_pairs,
        "corpus_duration_sec": corpus_duration_sec,
        "segment_target_seconds": args.segment_target_seconds,
        "segment_max_seconds": args.segment_max_seconds,
        "segment_min_seconds": args.segment_min_seconds,
        "dataset_jsonl": str(dataset_jsonl),
        "vectorized_dir": str(vectorized_dir),
        "training_dir": str(training_dir),
        "selected_checkpoint": str(selected_checkpoint),
        "serving_dir": str(serving_dir),
        "xcodec_checkpoint": str(xcodec_checkpoint),
        "prompt_wav": str(prompt_wav),
        "prompt_transcript_file": str(prompt_transcript_path),
        "text_file": str(text_file),
        "distributed_gpus": args.distributed_gpus,
        "train_epochs": effective_train_epochs,
        "val_epochs": effective_val_epochs,
        "config": config,
        "outputs": {},
    }

    for preset in selected_presets:
        output_path = outputs_dir / f"{preset.id}.wav"
        try:
            stats = synthesize_with_runtime(
                runtime=runtime,
                preset=preset,
                text_to_synthesize=text_to_synthesize,
                prompt_transcript=prompt_transcript,
                output_path=output_path,
            )
        except Exception as exc:
            summary["outputs"][preset.id] = {
                "path": str(output_path),
                "preset": dataclasses.asdict(preset),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[inworld-poc] synthesis failed for {preset.id}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        summary["outputs"][preset.id] = {
            "path": str(output_path),
            "preset": dataclasses.asdict(preset),
            **stats,
        }
        record_run(
            model_id=f"inworld_tts_poc_{preset.id}",
            output_path=output_path,
            input_path=text_file,
            reference_audio=prompt_wav if preset.mode == "clone" else None,
            manifest_path=args.manifest,
            extra={
                "cluster": resolve_cluster_name(),
                "base_model": args.base_model,
                "training_samples": len(corpus_entries),
                "train_epochs": effective_train_epochs,
                "codec_checkpoint": str(xcodec_checkpoint),
                "source_corpus": "qwen3_faster_clone_chunks",
                "preset": dataclasses.asdict(preset),
            },
        )

    ensure_parent(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[inworld-poc] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
