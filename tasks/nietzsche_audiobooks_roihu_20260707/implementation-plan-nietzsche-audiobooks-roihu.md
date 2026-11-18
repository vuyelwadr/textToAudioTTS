# Implementation Plan — Nietzsche Audiobooks on Roihu

## Context

- Existing template: `scripts/mahti/run_qwen_audiobook_video.sh` (used for 1984/Anna Karenina) already encodes the full recipe: `build_chunked_audiobook.py` with `qwen3_faster_clone` model + Brothers Karamazov reference voice, chunk ASR audit, then `make_autoscroll_video.py` with `--chapter-end-hold 3.0 --chapter-transition 0.6 --chapter-start-hold 2.0`.
- Roihu is CSC's new cluster (replacing Mahti/Puhti in 2026) and had never been used by this account before this task. It is architecturally different: GPU partitions (`gputest`/`gpuinteractive`/`gpumedium`/`gpularge`) run aarch64 Grace-Hopper GH200 nodes; CPU partitions (`test`/`small`/`medium`/`large`/...) run traditional x86_64 nodes. No Lmod module system is reachable on login or compute nodes.

## Decisions

- No YouTube download/ASR needed (per user correction) — source clean published text directly for all 4 books, then run the existing autoscroll-video pipeline exactly as done for 1984.
- Wire Roihu into the *existing* `scripts/mahti/*` cluster-parameterized scripts rather than creating a parallel `scripts/roihu/` tree (matches the README's own stated intent to keep these scripts cluster-portable ahead of the Mahti/Puhti retirement).
- Since Roihu has no module system, PyTorch is installed directly via pip (`--index-url https://download.pytorch.org/whl/cu128`, aarch64 manylinux wheels) instead of `module load pytorch`.
- Since Roihu's system Python is 3.9 (too old for `qwen-tts`/`faster-qwen3-tts`, which need >=3.10), provision a portable Python 3.11 via `uv` (no root needed) and prepend it to PATH before any venv creation.

## Code changes made

- `scripts/mahti/common.sh`: added `roihu` hostname detection to `detect_cluster_name()`; added `ensure_python310_plus()` (uv-provisioned Python 3.11 fallback) and `pip_install_pytorch_no_module()` (pip-based CUDA torch install, `--break-system-packages` since uv Pythons are PEP-668 externally-managed); wired both into `load_pytorch_module()`'s no-module branch.
- `scripts/mahti/common.py`: added `roihu` to the Python-side `resolve_cluster_name()` hostname fallback.
- `scripts/mahti/submit_model.sh`: added `roihu` cluster case with `gpu:gh200:1` GRES defaults (`gputest` smoke / `gpumedium` full).
- `scripts/mahti/submit_audiobook_video.sh`: added `roihu` cases for both `cpu` and `gpu` resource modes.

## Validation performed (real, on live Roihu account)

1. SSH discovery: confirmed hostname, SLURM partitions/GRES, project account access (`project_2010549` already has access to all Roihu partitions, no new allocation needed), aarch64 GH200 GPU driver 595.71.05 / CUDA 13.2, no ffmpeg/yt-dlp/module system on login.
2. Verified `pip install torch --index-url .../cu128` detects the GH200 GPU on a live `gputest` allocation.
3. Verified `imageio-ffmpeg` + `faster-whisper` work on the CPU (x86_64) `test` partition.
4. Hit and fixed two real environment bugs on the GPU (aarch64) path: system Python 3.9 too old for `qwen-tts`/`faster-qwen3-tts` (fixed via `uv`-provisioned Python 3.11), and PEP 668 externally-managed-environment block on the uv Python (fixed with `--break-system-packages`).
5. **End-to-end smoke test passed**: full `bootstrap_faster_qwen3_tts.sh` bootstrap + `run_faster_qwen3_tts.py --generation-mode clone` using the Brothers Karamazov reference produced a real, valid 12.88s/24kHz/mono WAV on a live `gpuinteractive` GH200 allocation. Verified locally via `ffprobe` (not just exit code).

## Remaining steps

- [ ] Source clean public-domain text for all 4 books (subagent in progress).
- [ ] Run one small preflight (2-3 chunks) of the full `run_qwen_audiobook_video.sh` recipe on Roihu end-to-end (chunk → TTS → CUDA audit → CPU video submit) before committing to full 4-8 hour runs.
- [ ] Submit full runs for all 4 books on Roihu.
- [ ] Verify each book's chunk audit pass rate and render the final MP4s with 1984 chapter-pause settings.
- [ ] rsync final MP4s + audio back to local library structure.

## Known non-blocking issue

- `sox` CLI binary is missing on Roihu (only the Python `sox` wrapper package installed); `faster-qwen3-tts` logs a warning and falls back gracefully — the smoke-test WAV was produced correctly despite it. Watch for this during full runs; revisit only if it causes an actual failure at scale.
