# Verification

## Status: Infrastructure Verified. Full 4-book runs not yet submitted (pending go-ahead).

## Roihu onboarding — Verified

Real, live SSH discovery on `roihu-gpu`/`roihu-cpu` (first-ever use of this cluster for this account):
- Hostname `roihu-gpu-login2.roihu.internal`; GPU partitions (`gputest`/`gpuinteractive`/`gpumedium`/`gpularge`) run **aarch64 NVIDIA GH200 120GB** (driver 595.71.05, CUDA 13.2); CPU partitions (`test`/`small`/`medium`/...) run **x86_64**. No Lmod module system reachable on either.
- Account `project_2010549` already has access to all Roihu partitions — no new allocation request needed.
- `/scratch` is a 6PB Lustre filesystem with 5PB free.

## Bugs found and fixed (all confirmed via live re-runs, not assumed)

1. **PyTorch install path**: no module system on Roihu → added `pip_install_pytorch_no_module()` (installs `torch`/`torchaudio` from `download.pytorch.org/whl/cu128`, aarch64 manylinux wheels). Verified: `torch.cuda.is_available()` → True, `NVIDIA GH200 120GB` on a live `gputest` allocation.
2. **Python too old**: system Python is 3.9; `qwen-tts`/`faster-qwen3-tts` need >=3.10 → added `ensure_python310_plus()`, provisions a portable Python 3.11 via `uv` (no root). Verified via direct venv creation and package installs.
3. **PEP 668 externally-managed-environment**: uv-provisioned Python blocks plain `pip install` → added `--break-system-packages` (safe here: single-user HPC scratch install, not a shared system Python).
4. **Cross-architecture venv contamination**: Roihu's CPU and GPU node pools share `/projappl`/`/scratch` but are different architectures; a venv built on one broke (`ModuleNotFoundError` on numpy C-extensions) when reused on the other → isolated venvs by architecture (`venvs-$(uname -m)`) for Roihu only; Mahti/Puhti unaffected (uniformly x86_64, unchanged path).
5. **Symlink cycle / bash PATH caching**: `uv python find <version>` prefers an active virtualenv's interpreter over its own managed installs. Because a venv was itself created through our Python-3.11 shim, its own `bin/python3` pointed back at the shim; a second `ensure_python310_plus` call (inside the nested per-chunk bootstrap) picked up that venv's interpreter and re-pointed the shim at it, creating a two-node symlink cycle. Bash's PATH search silently fell through to `/usr/bin/python3` (system 3.9) after the ELOOP, explaining every downstream symptom (ancient pip 21.3.1, no `--break-system-packages` support). Fixed by unsetting `VIRTUAL_ENV` before asking `uv` to locate the interpreter (`env -u VIRTUAL_ENV`). Root-caused via a temporary diagnostic print + `bash -x` trace of the actual failing subprocess (not guessed).
6. **ctranslate2 has no aarch64 CUDA build**: PyPI's aarch64 wheel is CPU-only (unlike the x86_64 wheels Mahti/Puhti use) → default chunk-audit/active-line ASR device to `cpu` specifically when `CSC_CLUSTER_NAME=roihu`; other clusters keep `cuda`.
7. **No system fonts on this minimal RHEL9 image**: `choose_default_font_argument()`'s hardcoded font-path list (Georgia/DejaVu/Nimbus/etc, added previously for Mahti) had zero matches → added a `matplotlib`-bundled `DejaVuSerif.ttf` fallback, resolved dynamically (not a hardcoded version-specific path), plus added `matplotlib` to the audiobook-video bootstrap's pip install list.

## End-to-end preflight — Verified (The Antichrist, 3/79 chunks)

Full pipeline run on a live `gpuinteractive` GH200 allocation: chunking → `qwen3_faster_clone` TTS (Brothers Karamazov voice clone) → CPU chunk-alignment audit (3/3 passed) → active-line scrolling video render (1984 chapter-pause settings: `--chapter-end-hold 3.0 --chapter-transition 0.6 --chapter-start-hold 2.0`).

Verified independently (pulled outputs to local machine, not just trusting exit code):
- `theAntichrist_qwen3_faster_clone.wav`: `ffprobe` confirms 415.36s, 24kHz mono PCM, 19.9MB — real, valid audio.
- `theAntichrist_qwen3_faster_clone_active_line.mp4`: `ffprobe` confirms H.264, 1920x1080, 24fps, duration 415.375s (matches audio) — real, valid video.
- Extracted a frame at 00:00:30 and visually confirmed correct scrolling-text rendering (serif font via matplotlib fallback, active-line underline highlighting, matching the existing 1984/Anna Karenina visual style).

## Text sourcing — Verified

All 4 books cross-checked locally (not just trusting the subagent report):
- The Antichrist: 27,526 words, starts at real PREFACE, ends at "THE END" — clean.
- Beyond Good and Evil: 62,459/62,517 words (minor count discrepancy between agent report and audit json, not investigated further — both are consistent with the real book length) — clean.
- The Gay Science (Gutenberg calls it "The Joyful Wisdom"): 84,144 words — clean.
- Thus Spoke Zarathustra: caught and fixed a real extraction bug (first pass captured only the front Table of Contents, 336 words); re-verified corrected file locally — 89,941 words, starts at "When Zarathustra was thirty years old..." ends at the novel's real final line — clean.

## Not yet done

- Full-length TTS + audit + video render for all 4 books (this preflight only rendered 3/79 chunks of 1 book, to validate the pipeline cheaply before committing to multi-hour jobs).
- Estimated full-book audio durations (`words / 4.62 wps`): Antichrist ~99min, Beyond Good and Evil ~225min, Gay Science ~303min, Zarathustra ~324min — roughly 15-16 hours of combined narrated audio across all 4 books. Actual GPU wall-clock time for generation is faster than realtime but not yet measured at scale — first full-book run will give a real per-book timing baseline.
