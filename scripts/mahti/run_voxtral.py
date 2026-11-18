from __future__ import annotations

import argparse
import base64
import io
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import requests
import soundfile as sf

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_model_root,
    resolve_output_dir,
    resolve_reference_text,
    safe_model_dirname,
)

MODEL_HUB_ID = "mistralai/Voxtral-4B-TTS-2603"
MODE_OUTPUTS = {
    "clone": "42_voxtral_4b_clone.wav",
    "plain": "41_voxtral_4b_plain.wav",
}
# Unique port avoids clashes with other vllm-omni servers
SERVER_PORT = 18603
SERVER_URL = f"http://localhost:{SERVER_PORT}"
SERVER_STARTUP_TIMEOUT_S = 600
SERVER_POLL_INTERVAL_S = 5
SERVER_LOG_TAIL_LINES = 200
DEFAULT_VOICE = "neutral_female"


@dataclass
class ServerProcess:
    proc: subprocess.Popen[str]
    log_tail: deque[str]
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Voxtral-4B-TTS-2603 via vllm-omni (plain preset-voice or voice-clone)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--generation-mode", choices=sorted(MODE_OUTPUTS), default="plain"
    )
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-text-file", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_HUB_ID)
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="Preset voice name for plain mode (e.g. neutral_female, cheerful_female).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def resolve_model_path(model_name: str) -> str:
    local_dir = resolve_model_root() / "voxtral" / safe_model_dirname(model_name)
    return str(local_dir if local_dir.exists() else model_name)


def resolve_stage_config_path() -> str:
    import vllm_omni.model_executor.stage_configs as stage_configs_pkg

    return str(Path(stage_configs_pkg.__file__).resolve().parent / "voxtral_tts.yaml")


def should_force_half_dtype() -> bool:
    try:
        import torch
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    major, _minor = torch.cuda.get_device_capability()
    return major < 8


def collect_server_output(stream: io.TextIOBase, log_tail: deque[str]) -> None:
    for line in stream:
        log_tail.append(line.rstrip())


def format_server_log_tail(log_tail: deque[str]) -> str:
    if not log_tail:
        return "(no server output captured)"
    return "\n".join(log_tail)


def start_server(model_path: str) -> ServerProcess:
    cmd = [
        "vllm-omni",
        "serve",
        model_path,
        "--stage-configs-path",
        resolve_stage_config_path(),
        "--omni",
        "--port",
        str(SERVER_PORT),
        "--trust-remote-code",
        "--enforce-eager",
    ]

    if should_force_half_dtype():
        cmd.extend(["--dtype", "half"])

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["VLLM_DISABLE_LOG_LOGO"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("failed to capture vllm server output")

    log_tail: deque[str] = deque(maxlen=SERVER_LOG_TAIL_LINES)
    stdout_thread = threading.Thread(
        target=collect_server_output, args=(proc.stdout, log_tail), daemon=True
    )
    stderr_thread = threading.Thread(
        target=collect_server_output, args=(proc.stderr, log_tail), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    return ServerProcess(
        proc=proc,
        log_tail=log_tail,
        stdout_thread=stdout_thread,
        stderr_thread=stderr_thread,
    )


def wait_for_server(server: ServerProcess, timeout_s: int) -> None:
    proc = server.proc
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "vllm server exited early with code "
                f"{proc.returncode}\nLast server output:\n{format_server_log_tail(server.log_tail)}"
            )
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(SERVER_POLL_INTERVAL_S)
    raise TimeoutError(
        "vllm server did not become ready within "
        f"{timeout_s}s\nLast server output:\n{format_server_log_tail(server.log_tail)}"
    )


def build_payload(args: argparse.Namespace, model_path: str, text: str) -> dict:
    payload: dict = {"model": model_path, "response_format": "wav"}
    if args.generation_mode == "plain":
        payload["input"] = text
        payload["voice"] = args.voice
    else:
        ref_text = resolve_reference_text(
            args.reference_audio,
            args.reference_text,
            args.reference_text_file,
        )
        payload["input"] = text
        payload["task_type"] = "Base"
        payload["ref_audio"] = base64.b64encode(
            args.reference_audio.read_bytes()
        ).decode()
        payload["ref_text"] = ref_text
    return payload


def main() -> None:
    args = parse_args()
    text = normalize_target_text(load_text(args.input))
    output_dir = resolve_output_dir(args.output_dir)
    output_path = args.output or (output_dir / MODE_OUTPUTS[args.generation_mode])
    ensure_parent(output_path)

    model_path = resolve_model_path(args.model_name)
    server = start_server(model_path)
    proc = server.proc
    try:
        wait_for_server(server, SERVER_STARTUP_TIMEOUT_S)

        payload = build_payload(args, model_path, text)
        resp = requests.post(f"{SERVER_URL}/v1/audio/speech", json=payload, timeout=300)
        resp.raise_for_status()

        audio_array, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        sf.write(str(output_path), audio_array, sr)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        server.stdout_thread.join(timeout=1)
        server.stderr_thread.join(timeout=1)

    reference_audio_path = (
        args.reference_audio if args.generation_mode == "clone" else None
    )
    record_run(
        model_id=f"voxtral_4b_{args.generation_mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=reference_audio_path,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "mode": args.generation_mode,
            "model_name": args.model_name,
            "voice": args.voice if args.generation_mode == "plain" else None,
            "server_port": SERVER_PORT,
        },
    )


if __name__ == "__main__":
    main()
