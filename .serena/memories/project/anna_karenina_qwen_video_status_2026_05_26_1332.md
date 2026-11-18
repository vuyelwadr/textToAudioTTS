# Anna Karenina Qwen status check - 2026-05-26 13:32 EEST

User asked to check whether both Anna Karenina runs are done and error-free, and to fix/rerun if needed.

Direct Mahti check:
- Full production GPU job `6720307` (`anna-qwen-full`) is still `PENDING` in `squeue`, reason `(Priority)`, elapsed `0:00`, time limit `1-12:00:00`.
- `sacct` state for `6720307`: `PENDING`, exit `0:0`, elapsed `00:00:00`, submit `2026-05-25T22:09:42`, start/end `Unknown`, no node assigned.
- `scontrol show job 6720307 -o`: partition `gpusmall`, account `project_2010549`, command `/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/tasks/anna_karenina_qwen_video_20260525/submit_anna_karenina_qwen_full_video.slurm`, stdout/stderr `slurm-anna-qwen-full-6720307.out`, resources `cpu=32,mem=96G,gres/gpu:a100=1,gres/nvme=900`; estimated `StartTime=2026-05-28T07:40:00`.
- Full runtime log `slurm-anna-qwen-full-6720307.out` does not exist yet, and full output root `generated/audiobooks-anna_karenina-qwen3_full_audited_20260525` does not exist yet. Expected because job has not started. No failure to fix/rerun.

Smoke validation:
- Smoke job `6720847` (`anna-qwen-smoke`) remains `COMPLETED`, exit `0:0`, elapsed `00:04:02`, node `g1101`.
- Smoke artifacts exist under `/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-anna_karenina-qwen3_smoke_2chunk_20260526`: two chunk WAVs, joined `AnnaKarenina_qwen3_faster_clone.wav`, `render_manifest.json`, `audiobook_summary.json`, and chunk audit.
- Smoke audit: `chunk_count=2`, `blocking_issue_count=0`, `trimmed_chunk_count=0`, `blocking_chunk_indices=[]`, `trimmed_chunk_indices=[]`.
- Smoke model log: `Batch complete: completed=2, failed=0`.
- Non-fatal warnings only: `sox` missing, NVML warning, deprecation warnings.

Updated and synced `tasks/anna_karenina_qwen_video_20260525/verification.md` to Mahti. Local and remote SHA-256: `1c19387457d1fd6868e571a409b1b823a8f22244f386b950a83416c65186c854`.

Current status wording: smoke run is Verified clean; full production run is queued/pending and not done, with no runtime errors because it has not started. No fix/rerun needed now.