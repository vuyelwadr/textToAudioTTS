# Anna Karenina Qwen Video Verification

Status: Production requeued after failed audit

## Local checks

- Extracted EPUB with `pandoc`, removed publisher/copyright front matter, started at `PART I`, and stopped after `THE END`.
- Cleaned source: `library/annaKarenina/AnnaKarenina.txt`
- Text audit: `library/annaKarenina/AnnaKarenina_text_audit.json`
- Dry chunk pass with 1984 settings produced `1139` chunks at `CHUNK_SIZE=1800`.
- Local and Mahti SHA-256 matched for the cleaned text, text audit, and Slurm script.

## Submission checks

- Uploaded cleaned source and task files to `/scratch/project_2010549/$USER/textToAudioTTS-mahti`.
- `sbatch --test-only` accepted the script.
- Submitted GPU job `6720307`.
- `squeue` verified `6720307|anna-qwen-full|PENDING|0:00|1-12:00:00|(Priority)`.
- `scontrol show job 6720307 -o` verified account `project_2010549`, partition `gpusmall`, command path, `cpu=32`, `mem=96G`, `gres/gpu:a100=1`, and `gres/nvme=900`.

## 2026-05-26 status check

- Full job `6720307` is still `PENDING` with reason `(Priority)`. It has not started, so there is no full-job runtime log or output root yet.
- Submitted smoke job `6720847` with the same Anna Karenina input, Qwen clone reference, `CHUNK_SIZE=1800`, audit settings, and `MAX_CHUNKS=2`.
- Smoke job `6720847` completed on Mahti `gputest` with state `COMPLETED` and exit `0:0`.
- Smoke output root: `/scratch/project_2010549/$USER/textToAudioTTS-mahti/generated/audiobooks-anna_karenina-qwen3_smoke_2chunk_20260526`.
- Smoke generated two chunk WAVs and joined audio:
  - `chunk_0000_5e619001.wav`
  - `chunk_0001_0becfcd5.wav`
  - `AnnaKarenina_qwen3_faster_clone.wav`
- Smoke chunk audit summary reported `chunk_count=2`, `blocking_issue_count=0`, `trimmed_chunk_count=0`, `blocking_chunk_indices=[]`, and `trimmed_chunk_indices=[]`.
- The smoke model log reported `Batch complete: completed=2, failed=0`.

## 2026-05-26 13:32 EEST progress check

- `squeue` verified full job `6720307|anna-qwen-full|PENDING|0:00|1-12:00:00|(Priority)`.
- `sacct` verified full job `6720307` is `PENDING`, exit `0:0`, elapsed `00:00:00`, start/end `Unknown`.
- `scontrol show job 6720307 -o` verified it is still pending for `Priority`, with no node allocated; Slurm estimated `StartTime=2026-05-28T07:40:00`.
- Full job log `slurm-anna-qwen-full-6720307.out` does not exist yet, and full output root `generated/audiobooks-anna_karenina-qwen3_full_audited_20260525` does not exist yet. This is expected because the job has not started.
- `sacct` verified smoke job `6720847` is `COMPLETED`, exit `0:0`, elapsed `00:04:02`, node `g1101`.
- Smoke artifacts were rechecked: two chunk WAVs exist, joined audio `AnnaKarenina_qwen3_faster_clone.wav` exists, `render_manifest.json` reports `status=ok`, and `audiobook_summary.json` points to the smoke audit.
- Smoke audit was rechecked: `chunk_count=2`, `blocking_issue_count=0`, `trimmed_chunk_count=0`, `blocking_chunk_indices=[]`, `trimmed_chunk_indices=[]`.
- Smoke model log was rechecked: `Batch complete: completed=2, failed=0`. The remaining messages are non-fatal environment warnings (`sox` missing, NVML warning, deprecation warnings).
- No fix or rerun was needed: the completed smoke run is clean, and the full production job has not failed because it has not started.

## 2026-05-26 13:49 EEST queue acceleration check

- Current full job already requests one GPU: `gres/gpu:a100:1,nvme:900`.
- `sprio -j 6720307` reported priority `307`; pending reason remains `Priority`, so the delay is scheduler priority, not a multi-GPU request.
- `sbatch --test-only` variants with the same script showed no earlier start for reduced walltimes or smaller resource shapes:
  - `36h`, `24h`, `20h`, `18h`, `12h`, `8h`, and `6h` all estimated `2026-05-28T04:59:44`.
  - Dropping NVMe from the test request did not improve the estimate.
  - Reducing to `16` CPUs and `64G` memory did not improve the estimate.
- Multi-GPU test requests were worse or disallowed:
  - `2` GPUs estimated `2026-05-28T22:19:24`, later than the one-GPU test estimate.
  - `4` GPUs failed test-only with `QOSMaxGRESPerJob`.
- Runtime estimate from the smoke model log:
  - First two chunks generated in `84.81s`, about `42.4s/chunk`, implying about `13.4h` for `1139` chunks if chunk time stays similar.
  - The chunk plan estimates `2329.5` minutes of audio; smoke RTF was about `0.40x`, implying about `15.4h` of TTS generation.
  - CPU TTS is not a practical queue workaround; even a conservative `5x-20x` CPU slowdown would put TTS alone around `2.8-11.2` days before audit/video overhead.
- Decision: keep the existing one-GPU job. Shortening resources does not improve the tested queue start, CPU is impractical, and multi-GPU either starts later or violates policy.

## Pending after queue

- Audio chunks, chunk audit, CPU video render, and final MP4 are not complete yet.
- After the GPU job runs, check the chunk audit summary before trusting the video submission.

## 2026-05-27 13:03 EEST production check and recovery

- Original full GPU job `6720307` ran from `2026-05-26T21:28:42` to `2026-05-27T10:18:13` and failed with exit `1:0`.
- Root cause was the required chunk audit, not Slurm quota or CUDA failure: `RuntimeError: Faster Qwen3-TTS clone chunk audit failed for chunk indices: 907`.
- Audit details for chunk `907`: status `truncated_current_chunk`, expected text coverage `0.645`, below the required `0.800`.
- A repair job `6731215` was submitted and split chunk `907` into four smaller prompts. It regenerated the four parts and rebuilt `chunk_0907_4b2b1309.wav`, then was cancelled at `00:07:12` after the generic wrapper began rerendering the whole model directory instead of only auditing/joining.
- Because that wrapper clears failed model state before rerendering, the failed-run chunk cache is no longer reusable. The original job was already failed and did not have a trusted final audiobook or video artifact.
- Replacement full GPU job `6731246` was submitted with one A100 in a fresh output root: `/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-anna_karenina-qwen3_full_rechunk1400_20260527`.
- Recovery settings: `CHUNK_SIZE=1400`, `AUDIT_CHUNKS=1`, `AUDIT_ASR_MODEL=small.en`, `AUDIT_ASR_DEVICE=cuda`, and `VIDEO_RENDER_MODE=submit_cpu`.
- `sbatch --test-only` accepted the replacement script and estimated `2026-05-29T13:58:45`; actual job `6731246` is currently `PENDING`.
