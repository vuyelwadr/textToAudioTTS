from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "tts_naturalness"
DEFAULT_INPUT = EXPERIMENT_ROOT / "inputs" / "naturalness_prompt.txt"
DEFAULT_REFERENCE_AUDIO = (
    REPO_ROOT / "assets" / "reference_audio" / "reference_voice.wav"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "generated" / "all_samples"
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "generated" / "runs" / "mahti_heavy_manifest.json"
DEFAULT_REFERENCE_TRANSCRIPT = DEFAULT_REFERENCE_AUDIO.with_suffix(".transcript.auto.txt")


def load_repo_env() -> None:
    env_path = REPO_ROOT / ".env"

    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)

    if os.environ.get("HUGGINGFACE_API_KEY"):
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.environ["HUGGINGFACE_API_KEY"])
        os.environ.setdefault("HF_TOKEN", os.environ["HUGGINGFACE_API_KEY"])
    elif os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ.setdefault("HF_TOKEN", os.environ["HUGGINGFACE_HUB_TOKEN"])
    elif os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", os.environ["HF_TOKEN"])


load_repo_env()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_visible_cuda_devices() -> list[str]:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw_value and raw_value.lower() != "all":
        return [
            device
            for device in (part.strip() for part in raw_value.split(","))
            if device and device.lower() not in {"none"}
        ]

    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    return [str(index) for index in range(torch.cuda.device_count())]


def resolve_cluster_name() -> str:
    for env_name in ("CSC_CLUSTER_NAME", "SLURM_CLUSTER_NAME"):
        value = os.environ.get(env_name)
        if value:
            lowered = value.lower()
            if "puhti" in lowered:
                return "puhti"
            if "mahti" in lowered:
                return "mahti"
            if "roihu" in lowered:
                return "roihu"
            return lowered

    hostname = socket.gethostname().lower()
    if "puhti" in hostname:
        return "puhti"
    if "mahti" in hostname:
        return "mahti"
    if "roihu" in hostname:
        return "roihu"
    return "local"


def resolve_torch_dtype(preferred: str | None = None) -> Any:
    try:
        import torch
    except ImportError:
        return None

    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if preferred:
        normalized = preferred.strip().lower()
        if normalized not in mapping:
            supported = ", ".join(sorted(mapping))
            raise ValueError(f"unsupported dtype {preferred!r}; expected one of: {supported}")
        return mapping[normalized]

    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16

    return torch.float32


def resolve_attn_implementation(preferred: str | None = None) -> str:
    if preferred:
        return preferred

    if resolve_device() != "cuda":
        return "eager"

    try:
        import importlib.util
        import torch
    except ImportError:
        return "sdpa"

    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        return "sdpa"

    if major >= 8 and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"

    return "sdpa"


def resolve_external_root() -> Path:
    return Path(os.environ.get("MAHTI_EXTERNAL_ROOT", REPO_ROOT / "external"))


def resolve_model_root() -> Path:
    return Path(os.environ.get("MAHTI_MODEL_ROOT", REPO_ROOT / "models"))


def resolve_output_dir(output_dir: str | None = None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        path = Path(os.environ.get("MAHTI_OUTPUT_ROOT", DEFAULT_OUTPUT_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_model_dirname(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def append_manifest(record: dict[str, Any], manifest_path: Path | None = None) -> None:
    manifest_file = manifest_path or DEFAULT_MANIFEST
    ensure_parent(manifest_file)

    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []

    existing.append(record)
    manifest_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def auto_transcribe_reference(audio_path: Path, output_path: Path | None = None) -> str:
    transcript_path = output_path or audio_path.with_suffix(".transcript.auto.txt")

    if transcript_path.exists():
        cached = load_text(transcript_path)
        if cached:
            return cached

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Reference text is required for this model. Install faster-whisper or pass --reference-text/--reference-text-file."
        ) from exc

    device = resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    model_name = os.environ.get("MAHTI_REFERENCE_ASR_MODEL", "small.en")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(audio_path), beam_size=5)
    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    if not transcript:
        raise RuntimeError(f"automatic transcription produced empty text for {audio_path}")

    write_text(transcript_path, transcript + "\n")
    return transcript


def resolve_reference_text(
    reference_audio: Path,
    reference_text: str | None,
    reference_text_file: Path | None,
) -> str:
    if reference_text:
        return reference_text.strip()
    if reference_text_file:
        return load_text(reference_text_file)
    return auto_transcribe_reference(reference_audio)


def normalize_target_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("input text is empty")
    return stripped


def manifest_path_value(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def record_run(
    *,
    model_id: str,
    output_path: Path,
    input_path: Path,
    reference_audio: Path | None,
    manifest_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "model_id": model_id,
        "output_path": manifest_path_value(output_path),
        "input_path": manifest_path_value(input_path),
    }
    if reference_audio is not None:
        record["reference_audio"] = manifest_path_value(reference_audio)
    if extra:
        record.update(extra)
    append_manifest(record, manifest_path=manifest_path)


def discover_first(base_dir: Path, *patterns: str) -> Path | None:
    for pattern in patterns:
        matches = sorted(base_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None
