# Hermetica batch — Talbot rerun + Divine Pymander + Kybalion (2026-11-19)

## Why

The 2026-11-14 `mysticismAndNewPhysics` run passed the old chunk audit but two
takes had run away to Qwen3-TTS's fixed 2048-token ceiling (163.84 s):

| chunk | audio | ASR score | coverage | position in joined audio |
|---|---|---|---|---|
| 14  | 163.84 s | 0.682 | 0.605 | ~9:37–12:20 (the audible break at ~10:31) |
| 214 | 163.84 s | 0.627 | 0.529 | later in the book |

The audit only treated low coverage as truncation when the transcript
self-match was high, so both slipped through. The audit now also blocks on:

- audio duration at/above `--decoder-ceiling-seconds` (default 158 s), and
- expected-word coverage below `--min-expected-ratio-hard` (default 0.65), and
- the minimum chunk F1 score default was raised from 0.55 to 0.70.

`build_chunked_audiobook.py` already had the self-heal loop (archive the failed
take, re-synthesize only the blocking chunk indices, re-audit, up to
`--max-heal-attempts`); the broader `blocking_chunk_indices` set now feeds it
automatically. `audit_summary_has_truncated_chunks()` uses the same rules so a
stale render can no longer be reused.

## Sources

- Talbot rerun: `library/mysticismAndNewPhysics/mysticismAndNewPhysics.txt`
  (existing text) into a fresh `full_20261119_v2` run.
- The Divine Pymander (Bodleian scan, CC BY-NC-SA 2.0):
  https://dbooks.bodleian.ox.ac.uk/books/PDFs/590481166.pdf
  - 153-page Google/Bodleian scan; no text layer. OCR'd with Tesseract 5.5.2
    (`tasks/hermetica_20261119/ocr_pymander.py`, 300 dpi, psm 3, 8 workers).
  - `tasks/hermetica_20261119/clean_pymander.py` removes page furniture,
    reconstructs the 17 books and numbered verses, and expands
    `Tut.`/`Herm.`/`Trism.` speaker labels for narration.
- The Kybalion, 1908 (YOGeBooks 2013 text-layer PDF):
  https://www.yogebooks.com/english/atkinson/1908kybalion.pdf
  - `tasks/hermetica_20261119/clean_kybalion.py` removes the YOGeBooks front
    matter, running heads and page numbers, rejoins hyphenation and drop caps,
    and stops at `Finis.`.

Prepared manuscripts: `library/divinePymander/divinePymander.txt` (32.4k words),
`library/kybalion/kybalion.txt` (33.8k words).

## Runs

All on Roihu, `gpumedium`, 2× GH200, 48 CPUs, `CHUNK_SIZE=850`,
`AUDIT_CHUNKS=1`, Schopenhauer `LhKBK6s_lMw` 10 s clone,
torch 2.9.1 pinned, video via `VIDEO_RENDER_MODE=submit_cpu` on Roihu `small`.

First attempt (2026-11-19 01:25):

| job | book | run label | job id | result |
|---|---|---|---|---|
| `talbot-rerun` | Mysticism and the New Physics | `full_20261119_v2` | 3217386 | TTS 511/511, audit crashed |
| `pymander-qwen` | The Divine Pymander | `full_20261119_v1` | 3217587 | TTS 242/242, audit crashed |
| `kybalion-qwen` | The Kybalion | `full_20261119_v1` | 3217388 | TTS + audit + heal OK, video failed |

Root cause: Roihu GPU nodes are aarch64 but CPU nodes are x86_64. The
`submit_audiobook_video.sh` CPU job inherited `MAHTI_ENV_ROOT=venvs-aarch64`
via `--export ALL`, installed x86_64 wheels into the aarch64 audit venv, and
corrupted it (`huggingface_hub`/`ffmpeg` missing). `run_audiobook_video.sh`
now unsets the inherited venv roots; the bootstrap pins
`huggingface_hub<1.0` + `faster-whisper==1.2.0` and validates the venv.

Second attempt (2026-11-19 ~02:50):

| job | purpose | job id |
|---|---|---|
| `talbot-resume` | reuse 511 chunk WAVs, audit + auto-heal 14/214, join, submit video | 3218804 |
| `kybalion-video` | reuse existing render, submit CPU video only | 3218805 |
| `pymander-v2` | fresh TTS/audit with the corrected opening sentence | 3218806 |

Outputs (Roihu):
`generated/audiobooks-mysticism_and_new_physics-full_20261119_v2/`
`generated/audiobooks-divine_pymander-full_20261119_v2/`
`generated/audiobooks-kybalion-full_20261119_v1/`

Outputs (Roihu):
`generated/audiobooks-mysticism_and_new_physics-full_20261119_v2/`
`generated/audiobooks-divine_pymander-full_20261119_v1/`
`generated/audiobooks-kybalion-full_20261119_v1/`

## Drive

Videos upload to the existing Google Drive `Books` folder
(`1hwBGIjlCFDitWVfP23-jkRyyPDeMGkRK`) using the Firefox-cookie resumable
uploader (`tasks/thought_forms_etidorhpa_20261113/upload_to_drive_books.py`).

New subfolders created 2026-11-19:

| book | folder id |
|---|---|
| The Divine Pymander | `1AlaX11Ui07jFjgo_lrIiDAaIM0_IjrMi` |
| The Kybalion | `14P0LHOGFbeBZmapwjKfFvI_eYNqVj-wE` |

`tasks/hermetica_20261119/upload_to_drive.py --folder-id ... --remote ...`
streams the mp4 from Roihu to Drive. Pass `--replace` to trash an existing
same-name file first (recoverable from Drive trash).

Uploaded:

| book | Drive file id | size |
|---|---|---|
| Mysticism and the New Physics | `1_yfp3PvbAJklakVdZWd6FCkX6Tm-aqW6` | 3,547,146,181 B |
| The Divine Pymander | `15M_Lyi_YFZLdRmu5ONr9RkHZ29TvON2o` | 1,669,962,760 B |
| The Kybalion | `1UjR0XeHmxdJ33ZOiDFq0aXS9r_j4fXtE` | 1,945,740,940 B |

The old broken Talbot file `1fFzVIpjj6bFlGk3orJ037wjydNeoLqvV`
(3,521,277,735 B) was moved to Drive trash, so it is recoverable if needed.
Each folder was re-listed after upload and contains exactly one file whose
name and byte size match the Roihu artifact.

## Verified done

- [x] Talbot chunk 14/214 auto-healed; 10:31 is now a normal chunk 15 take
      (40.96 s, ASR 0.950, coverage 0.955).
- [x] All three chunk audits report 0 blocking chunks.
- [x] Three 1080p24 MP4s rendered and ffprobe-checked (durations match the
      joined WAVs within the chapter-gap padding).
- [x] Three Drive files byte-size-verified in the correct Books subfolders.
- [x] Roihu aarch64/x86_64 venv split fixed and the audit stack pinned.
- [x] Code pushed to the `hermetica-20261119` branch.


Final render checks:

| book | duration | size | streams |
|---|---|---|---|
| Mysticism and the New Physics | 5:58:05 | 3.4 GB | 1080p24 + AAC 24 kHz mono |
| The Divine Pymander | 3:05:02 | 1.6 GB | 1080p24 + AAC 24 kHz mono |
| The Kybalion | 3:18:23 | 1.9 GB | 1080p24 + AAC 24 kHz mono |

The old 10:31 error is gone: the new Talbot chunk 14 is a normal 42.48 s
take, and 10:31 now falls inside chunk 15 (starts 10:23, 40.96 s, ASR score
0.950, coverage 0.955, status `ok`).


Note: the Drive name search did not consistently return the newly-created
empty subfolders, so the first Kybalion upload created a duplicate folder.
The file was moved into the original `14P0...` folder and the duplicate was
trashed; later uploads pass `--folder-id` explicitly.
