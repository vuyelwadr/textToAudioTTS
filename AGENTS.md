# Repo Agent Notes

Canonical global workflow: `/Users/ruwodda/.codex/instructions.md`

Project-specific rules for this repo:

- For long audiobook runs on CSC, keep GPU jobs focused on text chunking, TTS, and chunk audit. Do not render the final MP4 inline on a GPU job unless the user explicitly asks for that. Use `VIDEO_RENDER_MODE=submit_cpu`.
- Keep `AUDIT_CHUNKS=1` for production audiobook runs. Do not mark a run complete if the chunk audit reports blocking chunks.
- Keep `--sanitize-prechapter-note-blocks` enabled. It is the default guardrail against note/endnote walls being turned into audiobook chunks.
- For Qwen full-book runs, keep chunks small enough to avoid the model's fixed output ceiling. The production wrapper default is `CHUNK_SIZE=2200` and refuses values above `2600` unless `ALLOW_LARGE_QWEN_CHUNKS=1`; do not raise it without proving the chunk audit still passes.
- When generating or rerendering active-line video, reuse the chunk ASR cache with `--active-line-asr-cache-dir` whenever it exists.
- Prefer the cleaned `effective_input` and any `corrected_chunk_text_dir` over the raw source text when rendering video or debugging sync.
- After any text-cleaning or chunking-rule change, use a fresh output root instead of resuming on top of a known-bad run tree.
- Never call an audiobook run done until you have checked the actual artifact, the chunk audit summary, and the Slurm job state yourself.
