# Anna Karenina Qwen active-line video job - 2026-05-25

User asked to replicate the successful `library/1984` moving-cursor Qwen video settings for `library/annaKarenina` and queue the job on Mahti.

Local prep:
- Extracted `library/annaKarenina/Anna Karenina (Leo Tolstoy, Constance Garnett) (z-library.sk, 1lib.sk, z-lib.sk).epub` with pandoc to `library/annaKarenina/AnnaKarenina.raw.txt`.
- Cleaned to `library/annaKarenina/AnnaKarenina.txt`: starts at `PART I`, stops after `THE END`, removes pandoc `[]` placeholders and publisher/copyright front matter.
- Audit JSON: `library/annaKarenina/AnnaKarenina_text_audit.json`.
- Dry chunk check with 1984 settings (`CHUNK_SIZE=1800`, `MIN_CHUNK_SIZE=200`, `CHUNK_WORKERS=16`) produced 1139 chunks.

Run settings copied from 1984:
- model `qwen3_faster_clone`
- Brothers Karamazov 60s reference audio/transcript under `generated/clone_references_2026-04-25/`
- audit chunks enabled with `small.en` on CUDA and workers `auto`
- active-line video with ASR timing, `libx264`, preset `veryfast`, CRF `18`
- repo policy kept final MP4 render as `VIDEO_RENDER_MODE=submit_cpu`

Submitted Mahti GPU job:
- Job ID: `6720307`
- Name: `anna-qwen-full`
- Script: `/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/tasks/anna_karenina_qwen_video_20260525/submit_anna_karenina_qwen_full_video.slurm`
- Queue state at submission verification: PENDING, reason Priority
- GPU allocation: account `project_2010549`, partition `gpusmall`, time `1-12:00:00`, `cpu=32`, `mem=96G`, `gres/gpu:a100=1`, `gres/nvme=900`
- Output root: `/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-anna_karenina-qwen3_full_audited_20260525`

Do not mark complete until the Slurm state, chunk audit summary, and final MP4 artifact are checked.