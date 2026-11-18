#!/usr/bin/env python3
"""
Compatibility alias for scripts/build_scrolling_video.py.

The original multi-agent implementation plan for the unified scrolling-video
pipeline named the unified CLI entrypoint ``audiobook_video_pipeline.py``
(and ``scripts/mahti/submit_unified_pipeline.sh`` is specified to invoke it
under that name via ``submit_audiobook_video.sh --runner``). The actual
implementation task that built it was assigned the filename
``build_scrolling_video.py`` instead. This thin shim keeps both names
resolving to the exact same, single, real implementation, so callers written
against either path keep working.

Prefer running scripts/build_scrolling_video.py directly; this module exists
purely for path compatibility.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_scrolling_video import main, parse_args  # noqa: F401,E402

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    raise SystemExit(main())
