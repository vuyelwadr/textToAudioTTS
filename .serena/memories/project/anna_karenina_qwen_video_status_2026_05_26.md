# Anna Karenina Qwen status check - 2026-05-26

User asked to check job status and fix/rerun if errors.

Full job:
- Job `6720307` (`anna-qwen-full`) remains `PENDING` on Mahti, reason `(Priority)`, time limit `1-12:00:00`.
- `sacct` state `PENDING`, exit `0:0`, no node assigned, no runtime log yet, and full output root does not exist because it has not started.
- `scontrol` estimated start was `2026-05-30T13:40:00` at the time of check.

Smoke validation:
- Added/submitted `tasks/anna_karenina_qwen_video_20260525/smoke_anna_karenina_qwen_2chunk.slurm` to validate same source/model/reference/settings before full job starts.
- Smoke job `6720847` ran on Mahti `gputest`, state `COMPLETED`, exit `0:0`, elapsed `00:04:02`.
- Output root: `/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-anna_karenina-qwen3_smoke_2chunk_20260526`.
- Generated `chunk_0000_5e619001.wav`, `chunk_0001_0becfcd5.wav`, joined `AnnaKarenina_qwen3_faster_clone.wav`.
- Smoke audit: `chunk_count=2`, `blocking_issue_count=0`, `trimmed_chunk_count=0`, no blocking/trimmed indices.
- Model log: `Batch complete: completed=2, failed=0`.
- There was a non-fatal SoX warning during model load; job still completed successfully.

No fix/rerun of full job was needed because the full job has not failed and the smoke validation passed. Next check should revisit `6720307` after it starts/runs, then inspect its chunk audit and CPU video submission record.