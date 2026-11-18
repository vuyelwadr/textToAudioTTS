# User Messages

## 2026-05-26

The user first asked to use a YouTube Anna Karenina playlist as the audio source instead of TTS, extract audio and transcripts for four videos, render the same moving-cursor/active-line video style used for 1984, add correct chapter breaks if possible, produce four MP4 files, and join them into a fifth full MP4 if space permits.

The user then corrected the source playlist to:

`https://www.youtube.com/playlist?list=PLdk-d5njBw7laBIdDCpqxse1ToP9oNis1`

Requirements from the correction:

- Use only the Anna Karenina four parts in this playlist.
- Ignore/delete the earlier playlist choice.
- Use Session Context MCP to recover how existing Tolstoy YouTube items were processed.
- Prefer the existing local YouTube transcript/download workflow over online text searching.
- The local `library/tolstoy` folder already has YouTube videos with caption JSON and transcript files.
- The local full book may exist, but the important question is whether its text matches the YouTube video audio; YouTube-derived transcript text is the safer source for sync.

