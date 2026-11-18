#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import re
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "tts_naturalness"
DEFAULT_INPUT = EXPERIMENT_ROOT / "inputs" / "naturalness_prompt.txt"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "generated" / "all_samples"
DEFAULT_RUNS_DIR = EXPERIMENT_ROOT / "generated" / "runs"
DEFAULT_MATRIX_OUTPUT_DIR = EXPERIMENT_ROOT / "generated" / "all_samples" / "longform_benchmark"
DEFAULT_MATRIX_RUNS_DIR = EXPERIMENT_ROOT / "generated" / "runs" / "longform_benchmark"
DEFAULT_TOP_FOLDER = EXPERIMENT_ROOT / "generated" / "all_samples" / "top"
DEFAULT_CHATTERBOX_VOICE_SAMPLE = (
    REPO_ROOT
    / "engines"
    / "chatterbox-audio"
    / "voices"
    / "In_The_Shadow_Of_Fear_Rich_Miller_Direct.mp3"
)
LEGACY_KOKORO_PYTHON = REPO_ROOT / ".venv-kokoro-legacy" / "bin" / "python"
POCKET_TTS_PYTHON = REPO_ROOT / ".venv-pocket-tts" / "bin" / "python"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    rank: int
    label: str
    model_family: str
    output_name: str
    artifact_suffix: str
    kind: str
    notes: str
    kokoro_voice: str | None = None
    kokoro_release_id: str | None = None
    kokoro_voice_repo: str | None = None
    python_executable: str | None = None
    locale: str | None = None
    voice_name: str | None = None
    voice_filter: str | None = None
    default_enabled: bool = True

    @property
    def final_filename(self) -> str:
        return f"{self.rank:02d}_{self.output_name}{self.artifact_suffix}"


@dataclass(frozen=True)
class TopFinalistSpec:
    source_id: str
    rank: int
    label: str
    model_family: str
    venv_name: str
    runner_script: Path
    runner_args: tuple[str, ...]
    output_suffix: str
    notes: str

    @property
    def final_filename(self) -> str:
        return f"{self.rank:02d}_{self.source_id}{self.output_suffix}"


@dataclass(frozen=True)
class MatrixPassageSpec:
    passage_id: str
    order: int
    source_type: str
    source_path: Path
    label: str
    notes: str
    generated: bool
    word_count: int
    paragraph_count: int
    target_word_count: int | None = None


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_id="kokoro",
        rank=1,
        label="Kokoro 82M",
        model_family="Kokoro",
        output_name="kokoro_82m_af_alloy",
        artifact_suffix=".wav",
        kind="kokoro",
        notes="Local MPS-capable baseline using the repo's existing Kokoro script.",
        kokoro_voice="af_alloy",
    ),
    SourceSpec(
        source_id="kokoro_heart",
        rank=4,
        label="Kokoro 82M (af_heart)",
        model_family="Kokoro",
        output_name="kokoro_82m_af_heart",
        artifact_suffix=".wav",
        kind="kokoro",
        notes="Local Kokoro variant using the af_heart voice for same-model naturalness comparison.",
        kokoro_voice="af_heart",
        default_enabled=False,
    ),
    SourceSpec(
        source_id="kokoro_adam",
        rank=5,
        label="Kokoro 82M (am_adam)",
        model_family="Kokoro",
        output_name="kokoro_82m_am_adam",
        artifact_suffix=".wav",
        kind="kokoro",
        notes="Local Kokoro variant using the am_adam voice for same-model naturalness comparison.",
        kokoro_voice="am_adam",
        default_enabled=False,
    ),
    SourceSpec(
        source_id="kokoro_fable",
        rank=6,
        label="Kokoro 82M (bm_fable)",
        model_family="Kokoro",
        output_name="kokoro_82m_bm_fable",
        artifact_suffix=".wav",
        kind="kokoro",
        notes="Local Kokoro variant using the bm_fable voice for same-model naturalness comparison.",
        kokoro_voice="bm_fable",
        default_enabled=False,
    ),
    SourceSpec(
        source_id="kokoro_v1_0_alloy",
        rank=14,
        label="Kokoro v1.0 (af_alloy)",
        model_family="Kokoro",
        output_name="kokoro_v1_0_af_alloy",
        artifact_suffix=".wav",
        kind="kokoro_release",
        notes="Current Kokoro v1.0 weights using af_alloy for direct comparison against legacy v0.19 renders.",
        kokoro_voice="af_alloy",
        kokoro_release_id="v1_0",
        default_enabled=False,
    ),
    SourceSpec(
        source_id="kokoro_v0_19_alloy",
        rank=15,
        label="Kokoro v0.19 (af_alloy)",
        model_family="Kokoro Legacy",
        output_name="kokoro_v0_19_af_alloy",
        artifact_suffix=".wav",
        kind="kokoro_release",
        notes="Legacy Kokoro v0.19 weights rendered with the current af_alloy voice pack for direct listening comparison.",
        kokoro_voice="af_alloy",
        kokoro_release_id="v0_19",
        kokoro_voice_repo="hexgrad/Kokoro-82M",
        python_executable=str(LEGACY_KOKORO_PYTHON),
        default_enabled=False,
    ),
    SourceSpec(
        source_id="kokoro_v0_19_fp16_alloy",
        rank=16,
        label="Kokoro v0.19 fp16 (af_alloy)",
        model_family="Kokoro Legacy",
        output_name="kokoro_v0_19_fp16_af_alloy",
        artifact_suffix=".wav",
        kind="kokoro_release",
        notes="Legacy Kokoro v0.19 fp16 weights rendered with the current af_alloy voice pack.",
        kokoro_voice="af_alloy",
        kokoro_release_id="v0_19_fp16",
        kokoro_voice_repo="hexgrad/Kokoro-82M",
        python_executable=str(LEGACY_KOKORO_PYTHON),
        default_enabled=False,
    ),
    SourceSpec(
        source_id="chatterbox",
        rank=2,
        label="Chatterbox",
        model_family="Chatterbox",
        output_name="chatterbox_reference",
        artifact_suffix=".wav",
        kind="chatterbox",
        notes="Local Chatterbox run using the bundled engine script and reference sample.",
        default_enabled=False,
    ),
    SourceSpec(
        source_id="pocket_tts",
        rank=17,
        label="Pocket TTS (alba)",
        model_family="Pocket TTS",
        output_name="pocket_tts_alba",
        artifact_suffix=".wav",
        kind="pocket_tts",
        notes="Local CPU-first Pocket TTS baseline using Kyutai's 100M narration model and the built-in alba voice.",
        python_executable=str(POCKET_TTS_PYTHON),
        voice_name="alba",
    ),
    SourceSpec(
        source_id="tts_online",
        rank=3,
        label="text-to-speech.online",
        model_family="Edge-style web voice",
        output_name="text_to_speech_online_luke_neural",
        artifact_suffix=".mp3",
        kind="tts_online",
        notes="Browser-automated free web baseline using the repo's Playwright downloader.",
        locale="English (South Africa)",
        voice_name="LukeNeural(Male)",
        voice_filter="male",
    ),
    SourceSpec(
        source_id="orpheus_space",
        rank=11,
        label="Orpheus TTS (HF Space)",
        model_family="Orpheus",
        output_name="orpheus_tts_hf_space",
        artifact_suffix=".wav",
        kind="hf_space",
        notes="Remote public Gradio demo using a currently reachable Orpheus Space.",
    ),
    SourceSpec(
        source_id="fish_speech_space",
        rank=12,
        label="Fish Speech (HF Space)",
        model_family="Fish Speech",
        output_name="fish_speech_hf_space",
        artifact_suffix=".wav",
        kind="hf_space",
        notes="Remote public Gradio demo using a currently reachable Fish Speech Space.",
    ),
    SourceSpec(
        source_id="parler_tts_space",
        rank=13,
        label="Parler TTS (HF Space)",
        model_family="Parler TTS",
        output_name="parler_tts_hf_space",
        artifact_suffix=".wav",
        kind="hf_space",
        notes="Remote public Gradio demo using the public Parler TTS Space.",
    ),
    SourceSpec(
        source_id="tts_online_za_leah",
        rank=7,
        label="text-to-speech.online LeahNeural",
        model_family="Edge-style web voice",
        output_name="text_to_speech_online_leah_neural",
        artifact_suffix=".mp3",
        kind="tts_online",
        notes="Browser-automated free web baseline using the South Africa female voice.",
        locale="English (South Africa)",
        voice_name="LeahNeural(Female)",
        voice_filter="female",
        default_enabled=False,
    ),
    SourceSpec(
        source_id="tts_online_uk_ryan",
        rank=8,
        label="text-to-speech.online RyanNeural",
        model_family="Edge-style web voice",
        output_name="text_to_speech_online_ryan_neural_uk",
        artifact_suffix=".mp3",
        kind="tts_online",
        notes="Browser-automated free web baseline using the United Kingdom male voice.",
        locale="English (United Kingdom)",
        voice_name="RyanNeural(Male)",
        voice_filter="male",
        default_enabled=False,
    ),
    SourceSpec(
        source_id="tts_online_us_jenny",
        rank=9,
        label="text-to-speech.online JennyNeural",
        model_family="Edge-style web voice",
        output_name="text_to_speech_online_jenny_neural_us",
        artifact_suffix=".mp3",
        kind="tts_online",
        notes="Browser-automated free web baseline using the United States female voice.",
        locale="English (United States)",
        voice_name="JennyNeural(Female)",
        voice_filter="female",
        default_enabled=False,
    ),
    SourceSpec(
        source_id="tts_online_au_william",
        rank=10,
        label="text-to-speech.online WilliamNeural",
        model_family="Edge-style web voice",
        output_name="text_to_speech_online_william_neural_au",
        artifact_suffix=".mp3",
        kind="tts_online",
        notes="Browser-automated free web baseline using the Australia male voice.",
        locale="English (Australia)",
        voice_name="WilliamNeural(Male)",
        voice_filter="male",
        default_enabled=False,
    ),
)


TOP_FINALISTS: tuple[TopFinalistSpec, ...] = (
    TopFinalistSpec(
        source_id="faster_qwen3_tts_voice_clone_hacx",
        rank=63,
        label="Faster Qwen3-TTS voice clone",
        model_family="Faster Qwen3-TTS",
        venv_name="qwen3-tts-faster",
        runner_script=REPO_ROOT / "scripts" / "mahti" / "run_faster_qwen3_tts.py",
        runner_args=("--generation-mode", "clone"),
        output_suffix=".wav",
        notes="Current shortlist finalist from generated/all_samples/top.",
    ),
    TopFinalistSpec(
        source_id="voxcpm2_ultimate_clone",
        rank=75,
        label="VoxCPM2 ultimate clone",
        model_family="VoxCPM2",
        venv_name="voxcpm2",
        runner_script=REPO_ROOT / "scripts" / "mahti" / "run_voxcpm2.py",
        runner_args=("--generation-mode", "ultimate"),
        output_suffix=".wav",
        notes="Current shortlist finalist from generated/all_samples/top.",
    ),
    TopFinalistSpec(
        source_id="vibevoice_realtime_0_5b",
        rank=84,
        label="VibeVoice Realtime 0.5B",
        model_family="VibeVoice",
        venv_name="vibevoice",
        runner_script=REPO_ROOT / "scripts" / "mahti" / "run_vibevoice.py",
        runner_args=("--mode", "realtime"),
        output_suffix=".wav",
        notes="Current shortlist finalist from generated/all_samples/top.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repo's TTS naturalness comparison sources")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[source.source_id for source in SOURCES],
        default=[source.source_id for source in SOURCES if source.default_enabled],
        help="Subset of sources to run",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Shared comparison text file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Final sample folder")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="Per-run work folder")
    parser.add_argument(
        "--passage-manifest",
        type=Path,
        default=None,
        help="Combined long-form passage manifest for matrix generation",
    )
    parser.add_argument(
        "--top-folder",
        type=Path,
        default=DEFAULT_TOP_FOLDER,
        help="Folder containing the current shortlist finalists",
    )
    parser.add_argument(
        "--voice-sample",
        type=Path,
        default=DEFAULT_CHATTERBOX_VOICE_SAMPLE,
        help="Reference sample used for the Chatterbox run",
    )
    parser.add_argument("--web-visible", action="store_true", help="Show the browser for the web source")
    parser.add_argument("--force", action="store_true", help="Delete previous run folders for the selected sources")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse already generated final artifacts in the output folder instead of rerunning those sources",
    )
    return parser.parse_args()


def ensure_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def resolve_sources(selected_ids: Sequence[str]) -> list[SourceSpec]:
    wanted = set(selected_ids)
    return [source for source in SOURCES if source.source_id in wanted]


def build_command(source: SourceSpec, input_file: Path, work_dir: Path, args: argparse.Namespace) -> list[str]:
    python_executable = source.python_executable or sys.executable

    if source.kind == "kokoro":
        return [
            python_executable,
            str(REPO_ROOT / "scripts" / "audioChunkedKukuro.py"),
            str(input_file),
            "--voice",
            source.kokoro_voice or "af_alloy",
            "--output",
            str(work_dir),
            "--output-name",
            f"{source.source_id}.wav",
        ]

    if source.kind == "kokoro_release":
        command = [
            python_executable,
            str(REPO_ROOT / "scripts" / "generate_kokoro_release_sample.py"),
            str(input_file),
            "--release-id",
            source.kokoro_release_id or "v1_0",
            "--voice",
            source.kokoro_voice or "am_adam",
            "--output-dir",
            str(work_dir),
            "--output-name",
            f"{source.source_id}.wav",
        ]
        if source.kokoro_voice_repo:
            command.extend(["--voice-repo", source.kokoro_voice_repo])
        return command

    if source.kind == "chatterbox":
        return [
            python_executable,
            str(REPO_ROOT / "engines" / "chatterbox-audio" / "audioChunkedChatterboxGPU.py"),
            str(input_file),
            "--voice-sample",
            str(args.voice_sample),
            "--output-dir",
            str(work_dir),
        ]

    if source.kind == "pocket_tts":
        return [
            python_executable,
            str(REPO_ROOT / "scripts" / "generate_pocket_tts_sample.py"),
            str(input_file),
            "--voice",
            source.voice_name or "alba",
            "--output-dir",
            str(work_dir),
            "--output-name",
            f"{source.source_id}.wav",
        ]

    if source.kind == "tts_online":
        command = [
            python_executable,
            str(REPO_ROOT / "scripts" / "tts_download_production.py"),
            str(input_file),
            "-o",
            str(work_dir),
            "--locale",
            source.locale or "English (South Africa)",
            "--voice-name",
            source.voice_name or "LukeNeural(Male)",
            "--voice-filter",
            source.voice_filter or "male",
            "--require-exact-voice",
        ]
        if args.web_visible:
            command.append("--no-headless")
        return command

    if source.kind == "hf_space":
        return [
            python_executable,
            str(REPO_ROOT / "scripts" / "generate_hf_space_tts_sample.py"),
            str(input_file),
            "--source-id",
            source.source_id,
            "--output-dir",
            str(work_dir),
        ]

    raise ValueError(f"Unsupported source kind: {source.kind}")


def find_artifact(source: SourceSpec, work_dir: Path) -> Path:
    if source.kind in {"kokoro", "kokoro_release"}:
        expected = work_dir / f"{source.source_id}.wav"
        if expected.exists():
            return expected
        matches = sorted(work_dir.glob("*.wav"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"{source.source_id} produced {len(matches)} wav files instead of one final clip in {work_dir}"
            )

    if source.kind == "pocket_tts":
        expected = work_dir / f"{source.source_id}.wav"
        if expected.exists():
            return expected
        matches = sorted(work_dir.glob("*.wav"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"{source.source_id} produced multiple wav files in {work_dir}")

    if source.kind == "chatterbox":
        matches = sorted(work_dir.glob("chunk_*.wav"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"{source.source_id} produced {len(matches)} chunks; keep the prompt short or add clip assembly before promotion"
            )

    if source.kind == "tts_online":
        matches = sorted(work_dir.glob("chunk_*.mp3"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"{source.source_id} produced {len(matches)} chunks; keep the prompt short or add clip assembly before promotion"
            )

    if source.kind == "hf_space":
        preferred_matches = sorted(work_dir.glob(f"{source.source_id}.*"))
        audio_matches = [path for path in preferred_matches if path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}]
        if len(audio_matches) == 1:
            return audio_matches[0]
        matches = sorted(path for path in work_dir.iterdir() if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"{source.source_id} produced multiple audio files in {work_dir}")

    raise FileNotFoundError(f"No generated artifact found for {source.source_id} in {work_dir}")


def clean_work_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_log(work_dir: Path, completed: subprocess.CompletedProcess[str]) -> Path:
    log_path = work_dir / "run.log"
    combined_output = (completed.stdout or "")
    if completed.stderr:
        combined_output = f"{combined_output}\n\n[stderr]\n{completed.stderr}"
    log_path.write_text(combined_output.strip() + "\n", encoding="utf-8")
    return log_path


def copy_artifact(source: SourceSpec, artifact: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / source.final_filename
    shutil.copy2(artifact, final_path)
    return final_path


def build_reused_result(source: SourceSpec, final_path: Path) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "source_id": source.source_id,
        "label": source.label,
        "model_family": source.model_family,
        "notes": source.notes,
        "rank": source.rank,
        "command": [],
        "return_code": 0,
        "started_at": timestamp,
        "finished_at": timestamp,
        "log_path": None,
        "status": "ok",
        "artifact_path": str(final_path.relative_to(REPO_ROOT)),
        "artifact_size_bytes": final_path.stat().st_size,
        "reused_existing_artifact": True,
    }


def run_source(source: SourceSpec, args: argparse.Namespace) -> dict[str, object]:
    work_dir = args.runs_dir / source.source_id
    final_output_path = args.output_dir / source.final_filename

    if args.reuse_existing and not args.force and final_output_path.exists():
        return build_reused_result(source, final_output_path)

    if args.force:
        clean_work_dir(work_dir)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        for child in work_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    command = build_command(source, args.input, work_dir, args)
    command_env = os.environ.copy()
    if source.kind in {"kokoro", "kokoro_release"}:
        command_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    started_at = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=command_env,
    )
    finished_at = datetime.now(timezone.utc)
    log_path = write_log(work_dir, completed)

    result: dict[str, object] = {
        "source_id": source.source_id,
        "label": source.label,
        "model_family": source.model_family,
        "notes": source.notes,
        "rank": source.rank,
        "command": command,
        "return_code": completed.returncode,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "log_path": str(log_path.relative_to(REPO_ROOT)),
    }

    if completed.returncode != 0:
        result["status"] = "failed"
        result["error"] = f"Command exited with code {completed.returncode}"
        return result

    try:
        artifact = find_artifact(source, work_dir)
        final_path = copy_artifact(source, artifact, args.output_dir)
    except FileNotFoundError as error:
        result["status"] = "failed"
        result["error"] = str(error)
        return result

    result["status"] = "ok"
    result["artifact_path"] = str(final_path.relative_to(REPO_ROOT))
    result["artifact_size_bytes"] = final_path.stat().st_size
    return result


def write_manifest(args: argparse.Namespace, results: Iterable[dict[str, object]]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = args.runs_dir / f"run_manifest_{timestamp}.json"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.relative_to(REPO_ROOT)),
        "output_dir": str(args.output_dir.relative_to(REPO_ROOT)),
        "runs_dir": str(args.runs_dir.relative_to(REPO_ROOT)),
        "python_executable": sys.executable,
        "results": list(results),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    latest_manifest = args.runs_dir / "latest_manifest.json"
    latest_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def print_summary(results: Sequence[dict[str, object]], manifest_path: Path) -> None:
    print("\nTTS naturalness experiment summary")
    print("=" * 72)
    for result in results:
        label = result["label"]
        status = result["status"]
        if status == "ok":
            print(f"[ok] {label}: {result['artifact_path']}")
        else:
            print(f"[failed] {label}: {result.get('error', 'unknown error')}")
    print("-" * 72)
    print(f"Manifest: {manifest_path.relative_to(REPO_ROOT)}")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_matrix_passages(manifest_path: Path) -> list[MatrixPassageSpec]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("passages") or payload.get("generated_passages")
        if entries is None:
            raise ValueError(f"passage manifest does not contain entries: {manifest_path}")
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"unexpected passage manifest format in {manifest_path}")

    passages: list[MatrixPassageSpec] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"invalid passage entry at position {index} in {manifest_path}")

        source_path_value = entry.get("source_path") or entry.get("input_path") or entry.get("path")
        if not source_path_value:
            raise ValueError(f"passage entry {index} is missing a source path in {manifest_path}")

        source_path = resolve_repo_path(source_path_value)
        if not source_path.exists():
            raise FileNotFoundError(f"passage source file not found: {source_path}")

        normalized_text = normalize_target_text(load_text(source_path))
        paragraph_count = max(1, len([paragraph for paragraph in normalized_text.split("\n\n") if paragraph.strip()]))
        target_word_count = entry.get("target_word_count")
        passages.append(
            MatrixPassageSpec(
                passage_id=str(entry.get("passage_id") or source_path.stem),
                order=int(entry.get("order") or index),
                source_type=str(entry.get("source_type") or ("transcript_passage" if entry.get("generated") else "existing_input")),
                source_path=source_path,
                label=str(entry.get("label") or entry.get("role") or source_path.stem),
                notes=str(entry.get("notes") or ""),
                generated=bool(entry.get("generated", entry.get("source_type") == "transcript_passage")),
                word_count=int(entry.get("word_count") or count_words(normalized_text)),
                paragraph_count=int(entry.get("paragraph_count") or paragraph_count),
                target_word_count=int(target_word_count) if target_word_count is not None else None,
            )
        )

    return sorted(passages, key=lambda passage: (passage.order, passage.passage_id))


def discover_top_finalists(top_folder: Path) -> list[TopFinalistSpec]:
    if not top_folder.exists():
        raise FileNotFoundError(f"top folder not found: {top_folder}")

    finalists: list[TopFinalistSpec] = []
    missing: list[Path] = []
    for spec in TOP_FINALISTS:
        candidate = top_folder / spec.final_filename
        if candidate.exists():
            finalists.append(spec)
        else:
            missing.append(candidate)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"top folder is missing shortlisted finalists:\n{formatted}")

    return finalists


def build_top_command(spec: TopFinalistSpec, passage_input: Path, output_path: Path, manifest_path: Path) -> list[str]:
    script = shlex.quote(str(spec.runner_script.relative_to(REPO_ROOT)))
    input_arg = shlex.quote(str(passage_input.relative_to(REPO_ROOT)))
    output_arg = shlex.quote(str(output_path.relative_to(REPO_ROOT)))
    manifest_arg = shlex.quote(str(manifest_path.relative_to(REPO_ROOT)))
    extra_args = " ".join(shlex.quote(arg) for arg in spec.runner_args)
    command = (
        "set -euo pipefail && "
        "source scripts/mahti/common.sh && "
        f'if [[ -f "$MAHTI_ENV_ROOT/{spec.venv_name}/bin/activate" ]]; then '
        f"activate_python_venv {shlex.quote(spec.venv_name)}; "
        f"else ensure_python_venv {shlex.quote(spec.venv_name)}; fi && "
        f"python {script} --input {input_arg} --output {output_arg} --manifest {manifest_arg}"
    )
    if extra_args:
        command = f"{command} {extra_args}"
    return ["bash", "-lc", command]


def build_reused_top_result(
    spec: TopFinalistSpec,
    passage: MatrixPassageSpec,
    output_path: Path,
    manifest_path: Path,
    work_dir: Path,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "source_id": spec.source_id,
        "label": spec.label,
        "model_family": spec.model_family,
        "notes": spec.notes,
        "rank": spec.rank,
        "input_path": str(passage.source_path.relative_to(REPO_ROOT)),
        "output_path": str(output_path.relative_to(REPO_ROOT)),
        "command": [],
        "return_code": 0,
        "started_at": timestamp,
        "finished_at": timestamp,
        "log_path": str((work_dir / "run.log").relative_to(REPO_ROOT)) if (work_dir / "run.log").exists() else None,
        "status": "ok",
        "artifact_path": str(output_path.relative_to(REPO_ROOT)),
        "artifact_size_bytes": output_path.stat().st_size,
        "reused_existing_artifact": True,
        "passage_id": passage.passage_id,
        "passage_label": passage.label,
        "passage_order": passage.order,
        "passage_path": str(passage.source_path.relative_to(REPO_ROOT)),
        "passage_generated": passage.generated,
        "passage_word_count": passage.word_count,
        "passage_paragraph_count": passage.paragraph_count,
        "passage_target_word_count": passage.target_word_count,
        "runner_script": str(spec.runner_script.relative_to(REPO_ROOT)),
        "venv_name": spec.venv_name,
        "top_folder_entry": spec.final_filename,
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.exists() else None,
        "run_dir": str(work_dir.relative_to(REPO_ROOT)),
    }


def run_top_finalist(
    spec: TopFinalistSpec,
    passage: MatrixPassageSpec,
    output_dir: Path,
    runs_dir: Path,
    *,
    force: bool = False,
    reuse_existing: bool = False,
) -> dict[str, object]:
    passage_output_dir = output_dir / passage.passage_id
    work_dir = runs_dir / passage.passage_id / spec.source_id
    output_path = passage_output_dir / spec.final_filename
    manifest_path = work_dir / "manifest.json"

    if reuse_existing and not force and output_path.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        return build_reused_top_result(spec, passage, output_path, manifest_path, work_dir)

    if force:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        if output_path.exists():
            output_path.unlink()

    passage_output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    command = build_top_command(spec, passage.source_path, output_path, manifest_path)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    finished_at = datetime.now(timezone.utc)
    log_path = write_log(work_dir, completed)

    result: dict[str, object] = {
        "source_id": spec.source_id,
        "label": spec.label,
        "model_family": spec.model_family,
        "notes": spec.notes,
        "rank": spec.rank,
        "command": command,
        "return_code": completed.returncode,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "run_dir": str(work_dir.relative_to(REPO_ROOT)),
    }

    if completed.returncode != 0:
        result["status"] = "failed"
        result["error"] = f"Command exited with code {completed.returncode}"
        return result

    if not output_path.exists():
        result["status"] = "failed"
        result["error"] = f"Expected output not found at {output_path}"
        return result

    if not manifest_path.exists():
        result["status"] = "failed"
        result["error"] = f"Expected manifest not found at {manifest_path}"
        return result

    try:
        inner_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        result["status"] = "failed"
        result["error"] = f"Could not parse manifest {manifest_path}: {error}"
        return result

    inner_record = inner_manifest[-1] if isinstance(inner_manifest, list) and inner_manifest else inner_manifest
    if isinstance(inner_record, dict):
        for key, value in inner_record.items():
            result.setdefault(key, value)

    result.update(
        {
            "status": "ok",
            "input_path": str(passage.source_path.relative_to(REPO_ROOT)),
            "output_path": str(output_path.relative_to(REPO_ROOT)),
            "artifact_path": str(output_path.relative_to(REPO_ROOT)),
            "artifact_size_bytes": output_path.stat().st_size,
            "passage_id": passage.passage_id,
            "passage_label": passage.label,
            "passage_order": passage.order,
            "passage_path": str(passage.source_path.relative_to(REPO_ROOT)),
            "passage_generated": passage.generated,
            "passage_word_count": passage.word_count,
            "passage_paragraph_count": passage.paragraph_count,
            "passage_target_word_count": passage.target_word_count,
            "runner_script": str(spec.runner_script.relative_to(REPO_ROOT)),
            "venv_name": spec.venv_name,
            "top_folder_entry": spec.final_filename,
        }
    )
    if passage.target_word_count is not None:
        result["passage_target_word_count"] = passage.target_word_count
    return result


def run_matrix_mode(args: argparse.Namespace) -> int:
    passages = load_matrix_passages(args.passage_manifest)
    finalists = discover_top_finalists(args.top_folder)

    output_dir = args.output_dir
    runs_dir = args.runs_dir
    if output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = DEFAULT_MATRIX_OUTPUT_DIR
    if runs_dir == DEFAULT_RUNS_DIR:
        runs_dir = DEFAULT_MATRIX_RUNS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, object]] = []
    for passage in passages:
        passage_output_dir = output_dir / passage.passage_id
        passage_runs_dir = runs_dir / passage.passage_id
        passage_output_dir.mkdir(parents=True, exist_ok=True)
        passage_runs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Passage {passage.order:02d}: {passage.passage_id} ===")
        passage_results = [
            run_top_finalist(
                finalist,
                passage,
                output_dir,
                runs_dir,
                force=args.force,
                reuse_existing=args.reuse_existing,
            )
            for finalist in finalists
        ]
        passage_args = SimpleNamespace(input=passage.source_path, output_dir=passage_output_dir, runs_dir=passage_runs_dir)
        passage_manifest_path = write_manifest(passage_args, passage_results)
        print_summary(passage_results, passage_manifest_path)
        all_results.extend(passage_results)

    overall_args = SimpleNamespace(input=args.passage_manifest, output_dir=output_dir, runs_dir=runs_dir)
    overall_manifest_path = write_manifest(overall_args, all_results)
    print("\nLong-form benchmark complete")
    print(f"Manifest: {overall_manifest_path.relative_to(REPO_ROOT)}")

    return 0 if all(result["status"] == "ok" for result in all_results) else 1


def main() -> int:
    args = parse_args()
    if args.passage_manifest is not None:
        if not args.passage_manifest.exists():
            raise FileNotFoundError(f"passage manifest not found: {args.passage_manifest}")
        ensure_path_exists(args.top_folder, "Top finalist folder")
        return run_matrix_mode(args)

    selected_sources = resolve_sources(args.sources)
    ensure_path_exists(args.input, "Comparison input")
    ensure_path_exists(REPO_ROOT / "scripts" / "audioChunkedKukuro.py", "Kokoro script")
    ensure_path_exists(REPO_ROOT / "scripts" / "generate_kokoro_release_sample.py", "Kokoro release comparison script")
    ensure_path_exists(REPO_ROOT / "engines" / "chatterbox-audio" / "audioChunkedChatterboxGPU.py", "Chatterbox script")
    ensure_path_exists(REPO_ROOT / "scripts" / "generate_pocket_tts_sample.py", "Pocket TTS comparison script")
    ensure_path_exists(REPO_ROOT / "scripts" / "tts_download_production.py", "Web TTS script")
    ensure_path_exists(REPO_ROOT / "scripts" / "generate_hf_space_tts_sample.py", "HF Space TTS script")
    ensure_path_exists(args.voice_sample, "Chatterbox reference sample")

    if any(source.kind == "pocket_tts" for source in selected_sources):
        ensure_path_exists(POCKET_TTS_PYTHON, "Pocket TTS virtualenv Python")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    results = [run_source(source, args) for source in selected_sources]
    manifest_path = write_manifest(args, results)
    print_summary(results, manifest_path)

    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
