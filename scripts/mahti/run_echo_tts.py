"""Run Echo-TTS for the shared naturalness prompt.

Model   : jordand/echo-tts-base  (CC-BY-NC-SA-4.0 – non-commercial only)
GitHub  : https://github.com/jordandare/echo-tts
Blog    : https://jordandarefsky.com/blog/2025/echo/

Architecture overview
---------------------
Echo is a 2.4B diffusion transformer (DiT) conditioned on:
  • Target text  (byte-level tokeniser, max ~30 s worth of speech)
  • Speaker reference audio (up to 2 min; set to None for speaker-free mode)
It decodes through the Fish-Speech S1-DAC autoencoder to 44 100 Hz audio.

Inference path
--------------
1.  bootstrap_echo_tts.sh clones the GitHub repo to $MAHTI_EXTERNAL_ROOT/echo-tts
    and pre-downloads jordand/echo-tts-base into $MAHTI_MODEL_ROOT.
2.  This script injects a soundfile-backed torchcodec AudioDecoder shim,
    adds the repo root to sys.path, then imports inference.py for:
      load_model_from_hf, load_fish_ae_from_hf, load_pca_state_from_hf,
      sample_pipeline, sample_euler_cfg_independent_guidances
3.  Loads reference audio locally with soundfile, runs sample_pipeline(),
    and saves output with soundfile at 44 100 Hz.

Modes
-----
  plain  (78_echo_tts_base_plain.wav)  – no reference audio (speaker_audio=None)
  clone  (79_echo_tts_base_clone.wav)  – voice clone from --reference-audio

Recommended reference audio source (local, for pre-production testing):
  /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/outputs/
      video_downloads/haCxgJSFmFc/haCxgJSFmFc_reference_15s.wav

GPU: requires CUDA with ≥8 GB VRAM (A100 gives RTF < 0.05 at 30 s output).
"""

from __future__ import annotations

import argparse
import sys
import types
from functools import partial
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from common import (
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE_AUDIO,
    ensure_parent,
    load_text,
    normalize_target_text,
    record_run,
    resolve_cluster_name,
    resolve_external_root,
    resolve_output_dir,
)

DEFAULT_MODEL_REPO = "jordand/echo-tts-base"
DEFAULT_OUTPUT_PLAIN = "78_echo_tts_base_plain.wav"
DEFAULT_OUTPUT_CLONE = "79_echo_tts_base_clone.wav"
USER_REFERENCE_AUDIO = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "video_downloads"
    / "haCxgJSFmFc"
    / "haCxgJSFmFc_reference_8s.wav"
)
DEFAULT_ECHO_REFERENCE_AUDIO = (
    USER_REFERENCE_AUDIO if USER_REFERENCE_AUDIO.exists() else DEFAULT_REFERENCE_AUDIO
)


def install_torchcodec_audio_decoder_shim() -> None:
    """Provide the subset of torchcodec imported by Echo's inference module.

    Upstream inference.py imports torchcodec.AudioDecoder at module import time.
    On Puhti that native extension fails to load because libnvrtc.so.13 is
    absent, which blocks even plain-mode inference before the runner starts.
    """

    class _AudioMetadata:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = int(sample_rate)

    class _DecodedSamples:
        def __init__(self, data: torch.Tensor) -> None:
            self.data = data

    class _SoundFileAudioDecoder:
        def __init__(self, path: str | Path) -> None:
            data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
            self._waveform = torch.from_numpy(data.T.copy())
            self.metadata = _AudioMetadata(sample_rate)

        def get_samples_played_in_range(
            self,
            start_seconds: float,
            stop_seconds: float | None = None,
        ) -> _DecodedSamples:
            sample_rate = self.metadata.sample_rate
            start_frame = max(0, int(start_seconds * sample_rate))
            if stop_seconds is None:
                stop_frame = self._waveform.shape[1]
            else:
                stop_frame = min(
                    self._waveform.shape[1],
                    max(start_frame, int(stop_seconds * sample_rate)),
                )
            return _DecodedSamples(self._waveform[:, start_frame:stop_frame])

    decoders_module = types.ModuleType("torchcodec.decoders")
    decoders_module.__package__ = "torchcodec"
    decoders_module.AudioDecoder = _SoundFileAudioDecoder
    torchcodec_module = types.ModuleType("torchcodec")
    torchcodec_module.__path__ = []
    torchcodec_module.decoders = decoders_module
    sys.modules["torchcodec"] = torchcodec_module
    sys.modules["torchcodec.decoders"] = decoders_module


def load_reference_audio(
    path: Path,
    *,
    max_duration: int = 300,
    target_sample_rate: int = 44_100,
) -> torch.Tensor:
    """Load speaker reference audio without relying on torchcodec."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    waveform = waveform[:, : max_duration * sample_rate]
    if waveform.shape[1] == 0:
        raise ValueError(f"reference audio is empty: {path}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_sample_rate,
        )
    return waveform / waveform.abs().amax().clamp_min(1.0)

# Repo cloned by bootstrap_echo_tts.sh; model.py / autoencoder.py / inference.py
# / samplers.py all live here.
ECHO_ROOT = resolve_external_root() / "echo-tts"
if str(ECHO_ROOT) not in sys.path:
    sys.path.insert(0, str(ECHO_ROOT))

install_torchcodec_audio_decoder_shim()

# Deferred imports – only available after sys.path is patched above.
from inference import (  # noqa: E402  (module not in stdlib)
    load_fish_ae_from_hf,
    load_model_from_hf,
    load_pca_state_from_hf,
    sample_euler_cfg_independent_guidances,
    sample_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Echo-TTS for the shared naturalness prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--mode",
        choices=["plain", "clone"],
        default="plain",
        help=(
            "plain: generate without speaker reference (speaker_audio=None); "
            "clone: voice-clone from --reference-audio."
        ),
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=DEFAULT_ECHO_REFERENCE_AUDIO,
        help="Speaker reference WAV/MP3 (used in clone mode only).",
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=40,
        help="Diffusion steps (30 works well; 40 is quality default).",
    )
    parser.add_argument(
        "--cfg-text",
        type=float,
        default=3.0,
        help="Text CFG guidance scale.",
    )
    parser.add_argument(
        "--cfg-speaker",
        type=float,
        default=8.0,
        help="Speaker CFG guidance scale (ignored in plain mode).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=640,
        help=(
            "Latent sequence length: 640 ≈ 30 s; "
            "use 320 on low-VRAM (8 GB) cards."
        ),
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def build_sample_fn(args: argparse.Namespace):
    """Return a partial of sample_euler_cfg_independent_guidances (3× NFE)."""
    return partial(
        sample_euler_cfg_independent_guidances,
        num_steps=args.num_steps,
        cfg_scale_text=args.cfg_text,
        cfg_scale_speaker=args.cfg_speaker,
        # Apply CFG only in the middle of the diffusion trajectory.
        cfg_min_t=0.5,
        cfg_max_t=1.0,
        truncation_factor=None,
        rescale_k=None,
        rescale_sigma=None,
        speaker_kv_scale=None,
        speaker_kv_max_layers=None,
        speaker_kv_min_t=None,
        sequence_length=args.sequence_length,
    )


def prepare_audio_for_write(audio_out: torch.Tensor) -> torch.Tensor:
    """Normalize Echo output to a soundfile-compatible shape."""
    audio = audio_out.detach().float().cpu().squeeze()
    if audio.ndim == 0:
        raise ValueError("Echo-TTS returned a scalar instead of audio samples")
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return audio.transpose(0, 1).contiguous()
        return audio.contiguous()
    raise ValueError(f"unsupported Echo-TTS audio shape: {tuple(audio.shape)}")


def main() -> None:
    args = parse_args()

    text = normalize_target_text(load_text(args.input))
    # Echo-TTS expects speaker-tagged text; [S1] marks the primary speaker.
    text_prompt = f"[S1] {text}"

    output_dir = resolve_output_dir(args.output_dir)
    default_fname = (
        DEFAULT_OUTPUT_CLONE if args.mode == "clone" else DEFAULT_OUTPUT_PLAIN
    )
    output_path = args.output or (output_dir / default_fname)
    ensure_parent(output_path)

    # Load main DiT, Fish-Speech S1-DAC autoencoder, and PCA state.
    # All three download from HF hub on first use; subsequent runs resolve
    # from the cache at $MAHTI_CACHE_ROOT/huggingface/hub (set by common.sh).
    model = load_model_from_hf(
        repo_id=args.model_repo,
        device="cuda",
        dtype=None,  # defaults to bfloat16 inside load_model_from_hf
        delete_blockwise_modules=True,
    )
    fish_ae = load_fish_ae_from_hf()
    pca_state = load_pca_state_from_hf()

    if args.mode == "clone":
        ref_path: Path | None = args.reference_audio
        speaker_audio = load_reference_audio(ref_path).cuda()
    else:
        ref_path = None
        speaker_audio = None

    sample_fn = build_sample_fn(args)
    audio_out, _ = sample_pipeline(
        model=model,
        fish_ae=fish_ae,
        pca_state=pca_state,
        sample_fn=sample_fn,
        text_prompt=text_prompt,
        speaker_audio=speaker_audio,
        rng_seed=args.rng_seed,
    )

    # Echo-TTS outputs 44 100 Hz audio via the Fish-Speech S1-DAC codec.
    # Use soundfile to avoid torchaudio's torchcodec-backed save path on CSC.
    waveform = prepare_audio_for_write(audio_out).numpy()
    output_path.unlink(missing_ok=True)
    sf.write(str(output_path), waveform, 44100, format="WAV")

    record_run(
        model_id=f"echo_tts_base_{args.mode}",
        output_path=output_path,
        input_path=args.input,
        reference_audio=ref_path,
        manifest_path=args.manifest,
        extra={
            "cluster": resolve_cluster_name(),
            "model_repo": args.model_repo,
            "mode": args.mode,
            "num_steps": args.num_steps,
            "cfg_text": args.cfg_text,
            "cfg_speaker": args.cfg_speaker,
            "sequence_length": args.sequence_length,
            "rng_seed": args.rng_seed,
        },
    )


if __name__ == "__main__":
    main()
