# Half of a Yellow Sun Smoke Matrix

Purpose: compare short two-chunk audiobook smoke renders across multiple Qwen and
VibeVoice settings before committing to a full-book rerun.

Source passages:
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/source_two_chunks_narrative/chunk_0030_8c173cfe.txt`
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/source_two_chunks_narrative/chunk_0031_210766b6.txt`

These are narrative chunks from the Ugwu/Master section, not the book's front
matter blurbs.

Local root:
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/`

Qwen variants:
- `qwen_icl_det_ns`: full ICL clone, deterministic, non-streaming.
- `qwen_icl_det_stream`: full ICL clone, deterministic, streaming/default path.
- `qwen_xvec_det_ns`: x-vector-only clone baseline, deterministic, non-streaming.
- `qwen_icl_sample_low`: full ICL clone, sampled, `temperature=0.35`, `top_k=20`.
- `qwen_icl_sample_mid`: full ICL clone, sampled, `temperature=0.55`, `top_k=40`.

VibeVoice matrix:
- Speakers: `en-Mike_man`, `en-Carter_man`, `en-Davis_man`, `en-Emma_woman`, `en-Frank_man`, `en-Grace_woman`, `in-Samuel_man`.
- Per-speaker configs: `cfg13_s5`, `cfg15_s5`, `cfg17_s8`, `cfg20_s10`.
- Total VibeVoice variants: `28`.

Cluster split:
- Qwen sweep on `mahti`.
- VibeVoice sweep on `puhti`.

Machine-readable matrix:
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/variant_manifest.json`

Qwen local outputs:
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_icl_det_ns/chunks/chunk_0030_8c173cfe.wav` (`58.24s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_icl_det_ns/chunks/chunk_0031_210766b6.wav` (`44.80s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_icl_det_stream/chunks/chunk_0030_8c173cfe.wav` (`60.72s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_icl_det_stream/chunks/chunk_0031_210766b6.wav` (`45.52s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_icl_sample_low/chunks/chunk_0030_8c173cfe.wav` (`60.56s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_icl_sample_low/chunks/chunk_0031_210766b6.wav` (`43.12s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_icl_sample_mid/chunks/chunk_0030_8c173cfe.wav` (`61.60s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_icl_sample_mid/chunks/chunk_0031_210766b6.wav` (`43.04s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_xvec_det_ns/chunks/chunk_0030_8c173cfe.wav` (`63.36s`)
- `generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/qwen_xvec_det_ns/chunks/chunk_0031_210766b6.wav` (`43.84s`)

VibeVoice execution plan:
- The broken optimized batch path was replaced for the smoke matrix with per-file `scripts/mahti/run_vibevoice.py` realtime renders.
- The matrix runner now supports `--variant-ids` so speaker subsets can run as separate Puhti jobs against the same output tree.
- A first split-job wave exposed a realtime temp-file race in `scripts/mahti/run_vibevoice.py`: all parallel jobs shared the same scratch input/output stem under `cache/vibevoice/`.
- The runner now writes per-invocation scratch stems so parallel speaker jobs cannot steal each other's temporary files.
- Current clean Puhti wave:
- `en_mike_man`: `34065082`
- `en_carter_man`: `34065083`
- `en_davis_man`: `34065084`
- `en_emma_woman`: `34065085`
- `en_frank_man`: `34065086`
- `en_grace_woman`: `34065087`
- `in_samuel_man`: `34065088`
