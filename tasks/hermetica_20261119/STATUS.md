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
`AUDIT_CHUNKS=1`, `FORCE=1`, Schopenhauer `LhKBK6s_lMw` 10 s clone,
torch 2.9.1 pinned, video via `VIDEO_RENDER_MODE=submit_cpu` on Roihu `small`.

| job | book | run label | job id |
|---|---|---|---|
| `talbot-rerun` | Mysticism and the New Physics | `full_20261119_v2` | 3217386 |
| `pymander-qwen` | The Divine Pymander | `full_20261119_v1` | 3217387 |
| `kybalion-qwen` | The Kybalion | `full_20261119_v1` | 3217388 |

Outputs (Roihu):
`generated/audiobooks-mysticism_and_new_physics-full_20261119_v2/`
`generated/audiobooks-divine_pymander-full_20261119_v1/`
`generated/audiobooks-kybalion-full_20261119_v1/`

## Drive

Videos upload to the existing Google Drive `Books` folder
(`1hwBGIjlCFDitWVfP23-jkRyyPDeMGkRK`) using the Firefox-cookie resumable
uploader (`tasks/thought_forms_etidorhpa_20261113/upload_to_drive_books.py`).
