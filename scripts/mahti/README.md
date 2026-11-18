# CSC Heavy TTS

This folder wires heavyweight external TTS stacks into this repo so they can be bootstrapped and run on CSC GPU clusters instead of on the local Mac.

## Model IDs

- `orpheus`
- `fish-speech`
- `fish-s2pro`
- `f5-tts`
- `xtts`
- `gpt-sovits`
- `openvoice-v2`
- `omnivoice`
- `voxcpm2`
- `dia`
- `csm-1b`
- `qwen3-tts`
- `faster-qwen3-tts`
- `kyutai-tts`

## What The Scripts Do

- `common.sh`: shared CSC path, cache, Hugging Face, and virtualenv helpers
- `common.py`: shared prompt, reference audio, output directory, manifest, and reference transcription helpers
- `bootstrap_*.sh`: idempotent environment and checkpoint setup for each stack
- `run_*.py`: one-shot per-model generation entrypoints writing stable comparison files into `experiments/tts_naturalness/generated/all_samples/`
- `run_model.slurm`: generic Slurm entrypoint for `gputest` smoke jobs and cluster-specific full jobs
- `submit_model.sh`: helper that auto-detects Puhti vs Mahti, applies sane partition and runtime defaults, and forwards any runner flags, including `--mem` when one stack needs extra host RAM

## Defaults

- Shared prompt: `experiments/tts_naturalness/inputs/naturalness_prompt.txt`
- Shared reference audio: `assets/reference_audio/reference_voice.wav` (2.6s mono 24 kHz clip)
- Output folder: `experiments/tts_naturalness/generated/all_samples/`
- Manifest file: `experiments/tts_naturalness/generated/runs/mahti_heavy_manifest.json`

Default output filenames are intentionally stable so they drop into the existing listening pass alongside the earlier local and browser-based sources:

- `21_orpheus_tts_3b_leah.wav`
- `22_fish_speech_s2_pro.wav`
- `23_f5_tts_v1_base.wav`
- `24_xtts_v2.wav`
- `25_gpt_sovits_v2.wav`
- `26_openvoice_v2_en.wav`
- `27_omnivoice_clone.wav`
- `28_qwen3_tts_voice_design.wav`
- `29_faster_qwen3_tts_custom_voice.wav`
- `30_kyutai_tts_en.wav`
- `33_dia_1_6b_dialogue.wav`
- `32_csm_1b_plain.wav`
- `72_voxcpm2_tts.wav`
- `73_voxcpm2_voice_design.wav`
- `74_voxcpm2_clone.wav`
- `75_voxcpm2_ultimate_clone.wav`
- `77_dia_1_6b_clone.wav`
- `76_csm_1b_context_clone.wav`

## Cluster Defaults

- `submit_model.sh` auto-detects the cluster from `CSC_CLUSTER_NAME`, `SLURM_CLUSTER_NAME`, or the host name.
- On `mahti`, smoke runs default to `gputest` with `gpu:a100:1,nvme:900`, and full runs default to `gpusmall` with the same GPU class.
- On `puhti`, smoke runs default to `gputest` with `gpu:v100:1`, and full runs default to `gpu` with the same GPU class.
- Puhti GPU nodes used here are V100-based, while Mahti full GPU runs target A100 nodes with local NVMe. Use Puhti when you want to validate the V100 path explicitly, and Mahti when a stack benefits from newer A100 memory and NVMe.
- CSC has announced that both Puhti and Mahti retire in 2026 and will be replaced by Roihu. Keep the helper scripts cluster-parameterized instead of hard-coding one site-specific layout.
- Non-interactive Puhti sessions do not always pre-load `PROJAPPL` and `SCRATCH`, so `common.sh` now falls back to a repo-local `.csc/<cluster>/` layout unless you override the roots explicitly.

## First-Time Setup On CSC

Clone or rsync this repo into a project or scratch location on Puhti or Mahti. Then run one bootstrap per model you want to use.

The bootstrap scripts now serialize shared virtualenv setup and package installation, so multi-mode submissions for the same model can be launched in parallel without racing the same environment. If you want the cleanest path, pre-run the bootstrap once and then submit the actual Slurm jobs.

Examples:

```bash
cd /path/to/textToAudioTTS
bash scripts/mahti/bootstrap_orpheus.sh
bash scripts/mahti/bootstrap_fish_speech.sh
bash scripts/mahti/bootstrap_f5_tts.sh
bash scripts/mahti/bootstrap_xtts.sh
bash scripts/mahti/bootstrap_gpt_sovits.sh
bash scripts/mahti/bootstrap_openvoice_v2.sh
bash scripts/mahti/bootstrap_omnivoice.sh
bash scripts/mahti/bootstrap_voxcpm2.sh
bash scripts/mahti/bootstrap_dia.sh
bash scripts/mahti/bootstrap_csm_1b.sh
bash scripts/mahti/bootstrap_qwen3_tts.sh
bash scripts/mahti/bootstrap_faster_qwen3_tts.sh
bash scripts/mahti/bootstrap_kyutai_tts.sh
```

Notes:

- Orpheus now targets `canopylabs/orpheus-3b-0.1-ft`. Accept the model terms once on Hugging Face and keep the repo root `.env` populated with `HUGGINGFACE_API_KEY` so the CSC helpers can authenticate gated downloads.
- `common.sh` now auto-loads the repo root `.env` and maps `HUGGINGFACE_API_KEY` onto `HUGGINGFACE_HUB_TOKEN` and `HF_TOKEN` for `huggingface_hub` downloads.
- Fish Speech, F5-TTS, and XTTS v2 can download directly through the Hugging Face cache rooted under `MAHTI_CACHE_ROOT`. Fish Speech and F5-TTS now install a narrower inference-oriented dependency set so they reuse the CSC torch stack instead of redownloading a second one.
- GPT-SoVITS is the most fragile stack. The bootstrap script avoids the upstream conda installer, replaces the upstream forced `opencc` source build with a wheel-friendly substitute, and then downloads the pretrained bundle plus NLTK and Open JTalk assets.
- OpenVoice V2 pulls `checkpoints_v2_0417.zip` into the cloned repo and loads the companion MeloTTS source checkout directly from `MAHTI_EXTERNAL_ROOT/MeloTTS` so it does not need a separate editable Melo install.
- OmniVoice installs from PyPI and prefetches `k2-fsa/OmniVoice` into the shared model cache.
- VoxCPM2 installs from PyPI and prefetches `openbmb/VoxCPM2` into the shared model cache. The runner supports plain TTS, voice design, controllable cloning, and transcript-backed ultimate cloning.
- Dia installs through the Hugging Face `transformers` path and prefetches `nari-labs/Dia-1.6B-0626` into the shared model cache. The runner supports plain dialogue generation and audio-conditioned cloning against the shared reference clip.
- CSM-1B uses the official `transformers` integration and prefetches `sesame/csm-1b` into the shared model cache, so it is a good next-wave target when you want a high-quality open conversational model without a bespoke upstream repo.
- Qwen3-TTS installs from the `qwen-tts` package and prefetches the `1.7B-VoiceDesign` checkpoint used by the default runner.
- Faster Qwen3-TTS installs from the `faster-qwen3-tts` package and prefetches the `1.7B-CustomVoice` checkpoint used by the default runner.
- Kyutai TTS clones `kyutai-labs/delayed-streams-modeling` for its reference PyTorch script and installs `moshi==0.2.11` into the shared virtualenv.

## Running A Smoke Job

`gputest` is capped at 15 minutes and is intended only for validation.

Examples:

```bash
scripts/mahti/submit_model.sh orpheus --mode smoke --voice leah
scripts/mahti/submit_model.sh fish-speech --mode smoke
scripts/mahti/submit_model.sh f5-tts --mode smoke
scripts/mahti/submit_model.sh xtts --mode smoke --language en
scripts/mahti/submit_model.sh gpt-sovits --mode smoke --ref-language en --target-language en
scripts/mahti/submit_model.sh openvoice-v2 --mode smoke --language EN
scripts/mahti/submit_model.sh omnivoice --mode smoke --generation-mode clone --language English
scripts/mahti/submit_model.sh voxcpm2 --mode smoke --generation-mode tts
scripts/mahti/submit_model.sh dia --mode smoke --generation-mode dialogue
scripts/mahti/submit_model.sh csm-1b --mode smoke --generation-mode plain
scripts/mahti/submit_model.sh qwen3-tts --mode smoke --generation-mode design --language English
scripts/mahti/submit_model.sh faster-qwen3-tts --mode smoke --generation-mode custom --speaker Ryan --language English
scripts/mahti/submit_model.sh kyutai-tts --mode smoke
```

## Running A Full Job

Use `gpusmall` for actual generation runs.
For the Half of a Yellow Sun audiobook, the fastest path is to allocate two A100s and set `PARALLEL_MODELS=1` so Qwen3 and VibeVoice render at the same time.
Example:

```bash
PARALLEL_MODELS=1 FORCE=0 sbatch \
  --account project_2010549 \
  --partition gpusmall \
  --gres gpu:a100:2,nvme:900 \
  --cpus-per-task 32 \
  scripts/mahti/run_half_of_a_yellow_sun.sh
```

Examples:

```bash
scripts/mahti/submit_model.sh orpheus --mode full --voice leah
scripts/mahti/submit_model.sh fish-speech --mode full
scripts/mahti/submit_model.sh f5-tts --mode full --remove-silence
scripts/mahti/submit_model.sh xtts --mode full --language en
scripts/mahti/submit_model.sh gpt-sovits --mode full --ref-language en --target-language en
scripts/mahti/submit_model.sh openvoice-v2 --mode full --language EN --source-speaker EN-Default
scripts/mahti/submit_model.sh omnivoice --mode full --generation-mode clone --language English
scripts/mahti/submit_model.sh voxcpm2 --mode full --generation-mode clone
scripts/mahti/submit_model.sh dia --mode full --generation-mode clone
scripts/mahti/submit_model.sh csm-1b --mode full --generation-mode clone
scripts/mahti/submit_model.sh qwen3-tts --mode full --generation-mode design --language English
scripts/mahti/submit_model.sh faster-qwen3-tts --mode full --generation-mode custom --speaker Ryan --language English
scripts/mahti/submit_model.sh kyutai-tts --mode full
```

If the auto-detected cluster defaults are not what you want, override them explicitly with `--cluster`, `--partition`, `--cpus`, `--gres`, `--mem`, or `--time`.

## Audiobook Video Policy

For long audiobook runs, keep the GPU allocation focused on TTS plus chunk audit and submit the final MP4 render to a CPU partition. The repo wrappers now default to `VIDEO_RENDER_MODE=submit_cpu`, and the submitted video job reuses the chunk ASR cache from the audio audit when it is present.

If you need a standalone render for an existing audiobook summary, use the submit helper in CPU mode:

```bash
CSC_PROJECT=project_2010549 \
bash scripts/mahti/submit_audiobook_video.sh --cluster mahti --mode full --resource-mode cpu -- \
  --summary /scratch/project_2010549/$USER/textToAudioTTS-mahti/generated/.../audiobook_summary.json \
  --model qwen3_faster_clone \
  --show-active-line \
  --active-line-timing asr \
  --active-line-asr-device cpu \
  --active-line-asr-cache-dir /scratch/project_2010549/$USER/textToAudioTTS-mahti/generated/.../chunk_audit/asr_cache
```

## Monitoring Jobs And Logs

- Queue status for your jobs:

```bash
squeue -u "$USER"
```

- Accounting after a job starts or finishes:

```bash
sacct -j <jobid> --format=JobID,JobName%30,Partition,State,Elapsed,ExitCode
```

- Each submitted job writes a single combined stdout/stderr log into `experiments/tts_naturalness/generated/runs/slurm/` using the Slurm job id in the filename.

```bash
tail -f experiments/tts_naturalness/generated/runs/slurm/tts-puhti-<jobid>.out
```

- For bootstrap troubleshooting outside Slurm, keep using repo-local logs such as `logs/bootstrap_validation/*.log` so you can tell bootstrap failures apart from runner failures.

## Direct `sbatch` Usage

If you want full manual control, submit the generic Slurm entrypoint directly. On Mahti, keep using A100-style `--gres` values. On Puhti, switch them to `gpu:v100:1` and the `gpu` full partition.

```bash
sbatch \
  --account project_2000000 \
  --partition gpusmall \
  --time 02:00:00 \
  --cpus-per-task 32 \
  --gres gpu:a100:1,nvme:900 \
  scripts/mahti/run_model.slurm fish-speech --temperature 0.9
```

The `run_model.slurm` script expects the first positional argument to be one of the model ids listed above. Any remaining arguments are forwarded to the matching Python runner.

## Per-Model Notes

- `orpheus`: runner uses the repo-local `orpheus_tts_pypi` package path instead of trusting the published wheel, because the repo README explicitly calls out fixes that may not yet be on PyPI. The bootstrap and runner now default to `canopylabs/orpheus-3b-0.1-ft`.
- `fish-speech`: runner uses `--prompt-audio` and `--prompt-text` directly against `fish_speech/models/text2semantic/inference.py`, so it does not rely on a web server or space wrapper.
- `fish-s2pro`: dedicated wiring for the `fishaudio/s2-pro` checkpoint (Fish Audio Research License; free for research/non-commercial use, no HF gating encountered) using its own isolated `fish-s2pro` venv, but reusing the same `fish-speech` GitHub checkout and downloaded checkpoint directory as `fish-speech` above since it is the same upstream artifact. `run_fish_s2pro.py` matches the flat `--reference-audio`/`--reference-text-file`/`--input`/`--output`/`--device` CLI convention used by `faster-qwen3-tts`.
- `f5-tts`: if you leave reference text empty, F5-TTS transcribes it automatically. If you want to pin the transcript, pass `--reference-text` or `--reference-text-file`.
- `xtts`: simplest path here; only needs text plus a reference wav.
- `gpt-sovits`: runner writes temporary reference and target text files and shells into upstream `GPT_SoVITS/inference_cli.py`. English defaults are mapped to the CLI’s Chinese language labels, and the upstream stack expects a roughly 3-10 second reference clip.
- `openvoice-v2`: runner uses MeloTTS as the base speaker and then applies OpenVoice tone-color conversion against the repo reference audio.
- `omnivoice`: runner supports `clone`, `design`, and `auto` generation modes via `--generation-mode`. The default comparison path is `clone`, using the shared reference audio and transcript.
- `voxcpm2`: runner supports `tts`, `design`, `clone`, and `ultimate` generation modes via `--generation-mode`. `design` uses a natural-language voice description, `clone` conditions on a reference clip, and `ultimate` uses both the clip and its transcript for maximum similarity.
- `dia`: runner supports `dialogue` generation and `clone` generation via `--generation-mode`. The clone path uses Hugging Face’s audio-conditioned Dia pipeline and trims the prompt audio back off the decoded output so the saved file contains only the requested continuation.
- `csm-1b`: runner supports `plain` generation and a context-conditioned `clone` mode that anchors the target text to a reference audio/transcript turn. It uses the official Hugging Face `transformers` implementation, not a custom upstream inference stack.
- `qwen3-tts`: runner supports `design`, `custom`, and `clone` generation modes via `--generation-mode`. The default comparison path is `design`, using the `1.7B-VoiceDesign` checkpoint.
- `faster-qwen3-tts`: runner supports the same three generation modes via `--generation-mode` but requires CUDA. The default comparison path is `custom`, using the `1.7B-CustomVoice` checkpoint.
- `kyutai-tts`: runner shells into Kyutai’s upstream `scripts/tts_pytorch.py`, so its defaults stay close to the reference implementation.

## Environment Knobs

These helpers honor the following environment variables when you need to move caches or installs away from their defaults:

- `CSC_CLUSTER_NAME`
- `CSC_HEAVY_PROJECT_ROOT`
- `CSC_HEAVY_WORK_ROOT`
- `CSC_OUTPUT_ROOT`
- `MAHTI_PROJECT_ROOT`
- `MAHTI_WORK_ROOT`
- `MAHTI_EXTERNAL_ROOT`
- `MAHTI_ENV_ROOT`
- `MAHTI_MODEL_ROOT`
- `MAHTI_CACHE_ROOT`
- `MAHTI_OUTPUT_ROOT`
- `HF_HOME`
- `CSC_PROJECT`

If you are using `$LOCAL_SCRATCH`, the Slurm runner will expose it as `TMPDIR` automatically.

## Storage And Cleanup

- `common.sh` keeps long-lived installs under `MAHTI_PROJECT_ROOT` and puts run outputs, Hugging Face cache, pip cache, and temporary build files under the scratch-backed roots.
- `TMPDIR`, `TMP`, and `TEMP` are now redirected into scratch-backed storage so large pip downloads and wheel builds do not fill a small default temp filesystem.
- CSC is already warning that Puhti scratch is globally tight. Keep only active runs there; move anything long-lived into the project roots or delete it after validation.
- Quick size audit:

```bash
du -sh "$MAHTI_ENV_ROOT" "$MAHTI_MODEL_ROOT" "$MAHTI_CACHE_ROOT" "$MAHTI_OUTPUT_ROOT"
```

- Typical cleanup targets after a model is validated:

```bash
rm -rf "$MAHTI_CACHE_ROOT/tmp"
find experiments/tts_naturalness/generated/runs/slurm -type f -mtime +14 -delete
```
