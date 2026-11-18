## Dark Matter: Hidden Couplings

Found 9 file pairs that frequently co-change but have no import relationship:

| File A | File B | NPMI | Co-Changes | Lift |
|--------|--------|------|------------|------|
| docs/tts_takeover_context.md | scripts/mahti/run_model.slurm | 1.000 | 3 | 13.00 |
| docs/tts_takeover_context.md | scripts/mahti/submit_model.sh | 1.000 | 3 | 13.00 |
| scripts/mahti/run_model.slurm | scripts/mahti/submit_model.sh | 1.000 | 3 | 13.00 |
| docs/tts_takeover_context.md | scripts/mahti/common.sh | 0.888 | 3 | 9.75 |
| scripts/mahti/common.sh | scripts/mahti/run_model.slurm | 0.888 | 3 | 9.75 |
| scripts/mahti/common.sh | scripts/mahti/submit_model.sh | 0.888 | 3 | 9.75 |
| scripts/mahti/run_audiobook_video.py | scripts/make_autoscroll_video.py | 0.670 | 3 | 5.57 |
| scripts/build_chunked_audiobook.py | scripts/mahti/run_qwen3_batch_optimized.py | 0.598 | 4 | 3.90 |
| scripts/build_chunked_audiobook.py | scripts/mahti/run_vibevoice_batch_optimized.py | 0.531 | 3 | 3.90 |

These pairs likely share an architectural concern invisible to static analysis.
Consider adding explicit documentation or extracting the shared concern.