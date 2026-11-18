# TTS Naturalness Experiment

This workspace is the repo's clean comparison area for model-level TTS tests.

## Goal

- Compare the most natural sounding TTS models first, then let voice-level ranking happen by ear.
- Keep every generated comparison clip in one folder so you can cycle through them quickly.
- Reuse the repo's existing generation paths instead of building duplicate scripts.

## Current Scope

Implemented first-pass sources:

- `Kokoro 82M` via `scripts/audioChunkedKukuro.py`
- `Kokoro release comparisons` via `scripts/generate_kokoro_release_sample.py`
- `Chatterbox` via `engines/chatterbox-audio/audioChunkedChatterboxGPU.py`
- `text-to-speech.online` via `scripts/tts_download_production.py`
- `Orpheus TTS` via a public Hugging Face Space
- `Fish Speech` via a public Hugging Face Space
- `Parler TTS` via a public Hugging Face Space

Optional voice variants are still callable explicitly, but they are no longer part of the default comparison run:

- `Kokoro 82M`: `af_alloy`, `af_heart`, `am_adam`, `bm_fable`
- `Kokoro release comparisons`: `v1.0`, `v0.19`, `v0.19 fp16` using `af_alloy`
- `text-to-speech.online`: `LukeNeural`, `LeahNeural`, `RyanNeural`, `JennyNeural`, `WilliamNeural`

Current verified state on this machine:

- `Kokoro 82M`: verified through the shared experiment env with MPS plus CPU fallback for missing ops
- `Kokoro release comparisons`: verified for `v1.0`, `v0.19`, and `v0.19 fp16` using `af_alloy`
- `text-to-speech.online`: verified through the shared experiment env with exact voice locking
- `Orpheus TTS` via public Space: verified for one short comparison clip
- `Fish Speech` via public Space: verified for one short comparison clip
- `Parler TTS` via public Space: verified for one short comparison clip
- `Chatterbox`: code path updated and importable, but local model initialization is currently failing inside the shared env with `'NoneType' object is not callable`

Planned next sources:

- `Fish Speech` once the local editable repo is cloned and pinned cleanly into the comparison env
- `Bark / Coqui TTS` once the phase-2 dependency install is confirmed in the shared env
- `ElevenLabs` / `Cartesia` / `Google Speech Gen` if a stable free-tier or browser automation path is worth the effort

## Layout

- `inputs/naturalness_prompt.txt`: shared 15-second comparison text
- `generated/all_samples/`: the one folder for quick listening passes
- `generated/runs/`: per-source work dirs, logs, and manifests
- `model_rankings.md`: ranked shortlist of model families and priority notes

The shared prompt is intentionally kept short enough to stay single-chunk across the wired sources. If a source starts producing multiple chunks, the harness fails that run instead of silently copying only the first segment into the listening folder.

## Environment

Create the dedicated experiment environment once:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-tts-experiments
source .venv-tts-experiments/bin/activate
pip install --upgrade pip
pip install numpy==1.25.2
PIP_NO_BUILD_ISOLATION=1 pip install -r config/requirements.tts_experiments_env.txt
python -m playwright install chromium
```

Create the dedicated legacy Kokoro environment only if you want the official `v0.19` comparisons from `hexgrad/kLegacy`:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-kokoro-legacy
source .venv-kokoro-legacy/bin/activate
python -m pip install --upgrade pip
python -m pip install -r config/requirements.kokoro_legacy_env.txt
```

Online setup notes that were actually useful here:

- Current Kokoro `v1.0` setup is the official `pip install kokoro>=0.9.4 soundfile` path from `hexgrad/kokoro` plus `espeak-ng`.
- Legacy `v0.19` is not loaded by the current `KModel` path. Its official README uses the helper code shipped in `hexgrad/kLegacy/v0.19/` with `build_model(...)` from `models.py` and `generate(...)` from `kokoro.py`.
- Apple Silicon still benefits from `PYTORCH_ENABLE_MPS_FALLBACK=1` for both current and legacy Kokoro runs.

If you want to add Bark later from the same env, install Coqui TTS after the core comparison stack is stable:

```bash
source .venv-tts-experiments/bin/activate
pip install TTS==0.22.0
```

## Running The Comparison

Run every wired source:

```bash
source .venv-tts-experiments/bin/activate
python scripts/run_tts_naturalness_experiment.py
```

The default run is the distinct-source comparison set, not every optional voice variant.

Refresh the combined manifest from already generated core artifacts without rerunning slow remote demos:

```bash
source .venv-tts-experiments/bin/activate
python scripts/run_tts_naturalness_experiment.py --reuse-existing
```

Run a subset:

```bash
source .venv-tts-experiments/bin/activate
python scripts/run_tts_naturalness_experiment.py --sources kokoro chatterbox
```

Run the same-voice Kokoro release comparison set:

```bash
source .venv-tts-experiments/bin/activate
python scripts/run_tts_naturalness_experiment.py --sources kokoro_v1_0_alloy kokoro_v0_19_alloy kokoro_v0_19_fp16_alloy
```

Show the browser for the web source:

```bash
source .venv-tts-experiments/bin/activate
python scripts/run_tts_naturalness_experiment.py --sources tts_online --web-visible
```

## Output Naming

Files in `generated/all_samples/` are prefixed by ranking order and source name so they stay stable when you listen through them, for example:

- `01_kokoro_82m_af_alloy.wav`
- `03_text_to_speech_online_luke_neural.mp3`
- `11_orpheus_tts_hf_space.wav`
- `12_fish_speech_hf_space.wav`
- `13_parler_tts_hf_space.wav`
- `14_kokoro_v1_0_af_alloy.wav`
- `15_kokoro_v0_19_af_alloy.wav`
- `16_kokoro_v0_19_fp16_af_alloy.wav`

Each run also writes a manifest JSON file under `generated/runs/` with success or failure details for each source.

## CSC Heavy Models

The cluster-only heavyweight model workflow now lives under `scripts/mahti/`.

Use that path when you want repo-local setup and CSC execution for external heavyweight stacks such as Orpheus, Fish Speech, F5-TTS, XTTS v2, GPT-SoVITS, OpenVoice V2, OmniVoice, Qwen3-TTS, Faster Qwen3-TTS, and Kyutai TTS.

The runbook is in `scripts/mahti/README.md`. The helper now auto-detects Mahti vs Puhti, and the default CSC comparison outputs are written into this same experiment area under `generated/all_samples/` using the next stable ranking slots (`21_` through `30_`).