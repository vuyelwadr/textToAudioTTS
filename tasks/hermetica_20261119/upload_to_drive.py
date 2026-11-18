#!/usr/bin/env python3
"""Create a Drive Books subfolder (if missing) and upload Roihu mp4s into it.

Reuses the Firefox-cookie Drive client from the 2026-11-13 books task.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tasks" / "thought_forms_etidorhpa_20261113"))

from upload_to_drive_books import (  # noqa: E402
    DRIVE_PUBLIC_KEY,
    DriveClient,
    ensure_remote_uploader,
    upload_remote_file,
)


def ensure_folder(drive: DriveClient, name: str, parent_id: str) -> str:
    existing = drive.find_file_in_folder(name, parent_id)
    if existing and existing.get("mimeType", "").endswith("folder"):
        print(f"[skip] folder {name!r} already exists id={existing['id']}")
        return existing["id"]
    url = f"https://clients6.google.com/drive/v3/files"
    params = {"key": DRIVE_PUBLIC_KEY, "supportsAllDrives": "true", "fields": "id,name"}
    headers = drive.api_headers()
    headers["Content-Type"] = "application/json; charset=UTF-8"
    body = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    for attempt in range(2):
        resp = drive.session.post(url, params=params, headers=headers, data=json.dumps(body), timeout=60)
        if resp.status_code == 401 and attempt == 0:
            drive.refresh_cookies()
            headers = drive.api_headers()
            headers["Content-Type"] = "application/json; charset=UTF-8"
            continue
        resp.raise_for_status()
        created = resp.json()
        print(f"[ok] created folder {name!r} id={created['id']}")
        return created["id"]
    raise RuntimeError(f"could not create Drive folder {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="1hwBGIjlCFDitWVfP23-jkRyyPDeMGkRK", help="Drive Books folder id")
    parser.add_argument("--folder-name", required=True)
    parser.add_argument("--remote", action="append", default=[], help="Absolute Roihu file path")
    parser.add_argument("--authuser", type=int, default=0)
    args = parser.parse_args()

    ensure_remote_uploader()
    drive = DriveClient(authuser=args.authuser)
    folder_id = ensure_folder(drive, args.folder_name, args.parent)
    for remote in args.remote:
        upload_remote_file(remote, folder_id, drive)
    print(f"folder_id={folder_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
