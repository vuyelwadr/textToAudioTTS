#!/usr/bin/env python3
"""Extract repo-scoped user messages from local Codex, Copilot CLI, and Claude transcripts.

Only sessions whose recorded workspace or cwd exactly matches this repository's
absolute path are included.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from typing import Any
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_PATH = str(REPO_ROOT.resolve())

COPILOT_WORKSPACE_STORES = [
    Path.home() / "Library/Application Support/Code/User/workspaceStorage",
    Path.home() / "Library/Application Support/Code - Insiders/User/workspaceStorage",
]
COPILOT_CLI_SESSIONS_ROOT = Path.home() / ".copilot" / "session-state"
CODEX_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"
CLAUDE_SESSIONS_ROOT = Path.home() / ".claude" / "projects"
SOURCE_ORDER = ["codex", "claude", "copilot", "copilot-cli"]


def _normalize_repo_reference(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        raw = unquote(parsed.path or raw.removeprefix("file://"))
    try:
        return str(Path(raw).expanduser().resolve())
    except OSError:
        return ""


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(
            part.strip() for part in parts if part and part.strip()
        ).strip()
    return ""


def _flatten_codex_content(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _flatten_claude_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _file_mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return "unknown"


def _timestamp_sort_key(timestamp: str | None, fallback: str) -> tuple[str, str]:
    return (timestamp or "", fallback)


def _is_real_codex_user_message(text: str) -> bool:
    if not text:
        return False
    return not text.startswith("# AGENTS.md instructions for ")


def _is_real_claude_user_message(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("<local-command-"):
        return False
    if lowered.startswith("<command-name>"):
        return False
    return True


def _prefilter_candidate_files(root: Path, needle: str) -> list[Path]:
    if not root.is_dir():
        return []
    rg = shutil.which("rg")
    if rg:
        try:
            proc = subprocess.run(
                [rg, "-l", "--fixed-strings", needle, str(root)],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            proc = None
        if proc and proc.returncode in {0, 1}:
            return [
                Path(line)
                for line in proc.stdout.splitlines()
                if line.strip().endswith(".jsonl")
            ]
    return sorted(root.rglob("*.jsonl"))


def find_copilot_transcript_dirs() -> list[Path]:
    transcript_dirs: list[Path] = []
    seen: set[Path] = set()
    for store_root in COPILOT_WORKSPACE_STORES:
        if not store_root.is_dir():
            continue
        for workspace_dir in store_root.iterdir():
            workspace_json = workspace_dir / "workspace.json"
            if not workspace_json.is_file():
                continue
            try:
                data = json.loads(workspace_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            folder = data.get("folder")
            if _normalize_repo_reference(folder) != REPO_PATH:
                continue
            transcripts_dir = workspace_dir / "GitHub.copilot-chat" / "transcripts"
            if transcripts_dir.is_dir() and transcripts_dir not in seen:
                transcript_dirs.append(transcripts_dir)
                seen.add(transcripts_dir)
    return transcript_dirs


def extract_copilot_messages() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for transcripts_dir in find_copilot_transcript_dirs():
        for jsonl_file in sorted(
            transcripts_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime
        ):
            session_id = jsonl_file.stem
            session_modified = _file_mtime_iso(jsonl_file)
            try:
                lines = jsonl_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                entry_type = entry.get("type", "")
                data = entry.get("data", {})
                timestamp = entry.get("timestamp")
                if entry_type == "user.message":
                    text = _flatten_text(data.get("content", ""))
                elif entry_type == "conversation.summary":
                    summary = _flatten_text(data.get("content", ""))
                    text = f"[COMPACTED SUMMARY]\n{summary}" if summary else ""
                else:
                    continue
                if not text:
                    continue
                messages.append(
                    {
                        "source": "copilot",
                        "session_id": session_id,
                        "session_path": str(jsonl_file),
                        "session_modified": session_modified,
                        "line_no": line_no,
                        "timestamp": timestamp,
                        "text": text,
                    }
                )
    return messages


def extract_copilot_cli_messages() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if not COPILOT_CLI_SESSIONS_ROOT.is_dir():
        return messages
    for session_dir in COPILOT_CLI_SESSIONS_ROOT.iterdir():
        if not session_dir.is_dir():
            continue
        events_file = session_dir / "events.jsonl"
        if not events_file.is_file():
            continue
        session_data: dict[str, Any] = {}
        meta_file = session_dir / "session.metadata.json"
        if meta_file.is_file():
            try:
                session_data = json.loads(meta_file.read_text())
            except (OSError, json.JSONDecodeError):
                session_data = {}
        if not session_data.get("cwd"):
            workspace_file = session_dir / "workspace.yaml"
            if workspace_file.is_file():
                try:
                    import yaml as _yaml

                    session_data = _yaml.safe_load(workspace_file.read_text()) or {}
                except Exception:
                    session_data = {}
                    for line in workspace_file.read_text().splitlines():
                        line = line.strip()
                        if ":" in line and not line.startswith("#"):
                            key, _, val = line.partition(":")
                            session_data[key.strip()] = val.strip()
        cwd = session_data.get("cwd", "")
        if _normalize_repo_reference(cwd) != REPO_PATH:
            continue
        session_id = session_dir.name
        session_modified = _file_mtime_iso(events_file)
        try:
            lines = events_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "user.message":
                continue
            data = entry.get("data", {})
            if isinstance(data, dict):
                text = data.get("content", "")
            elif isinstance(data, str):
                text = data
            else:
                text = ""
            if not isinstance(text, str):
                text = ""
            text = text.strip()
            if not text:
                continue
            messages.append(
                {
                    "source": "copilot-cli",
                    "session_id": session_id,
                    "session_path": str(events_file),
                    "session_modified": session_modified,
                    "line_no": line_no,
                    "timestamp": entry.get("timestamp"),
                    "text": text,
                }
            )
    return messages


def extract_codex_messages() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if not CODEX_SESSIONS_ROOT.is_dir():
        return messages
    candidate_files = _prefilter_candidate_files(CODEX_SESSIONS_ROOT, REPO_PATH)
    for jsonl_file in sorted(candidate_files):
        try:
            lines = jsonl_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        session_id = jsonl_file.stem
        repo_match = False
        for line in lines[:5]:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "session_meta":
                continue
            payload = entry.get("payload") or {}
            session_id = payload.get("id") or session_id
            repo_match = _normalize_repo_reference(payload.get("cwd")) == REPO_PATH
            break
        if not repo_match:
            continue
        session_modified = _file_mtime_iso(jsonl_file)
        for line_no, line in enumerate(lines, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "response_item":
                continue
            payload = entry.get("payload") or {}
            if payload.get("type") != "message" or payload.get("role") != "user":
                continue
            text = _flatten_codex_content(payload.get("content"))
            if not _is_real_codex_user_message(text):
                continue
            messages.append(
                {
                    "source": "codex",
                    "session_id": session_id,
                    "session_path": str(jsonl_file),
                    "session_modified": session_modified,
                    "line_no": line_no,
                    "timestamp": entry.get("timestamp"),
                    "text": text,
                }
            )
    return messages


def extract_claude_messages() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if not CLAUDE_SESSIONS_ROOT.is_dir():
        return messages
    candidate_files = _prefilter_candidate_files(CLAUDE_SESSIONS_ROOT, REPO_PATH)
    for jsonl_file in sorted(candidate_files):
        try:
            lines = jsonl_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        session_rows: list[tuple[int, dict[str, Any]]] = []
        session_id = jsonl_file.stem
        for line_no, line in enumerate(lines, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _normalize_repo_reference(entry.get("cwd")) != REPO_PATH:
                continue
            if isinstance(entry.get("sessionId"), str) and entry["sessionId"].strip():
                session_id = entry["sessionId"].strip()
            session_rows.append((line_no, entry))
        if not session_rows:
            continue
        session_modified = _file_mtime_iso(jsonl_file)
        for line_no, entry in session_rows:
            if entry.get("type") != "user" or entry.get("isMeta"):
                continue
            message = entry.get("message") or {}
            if message.get("role") != "user":
                continue
            text = _flatten_claude_content(message.get("content"))
            if not _is_real_claude_user_message(text):
                continue
            messages.append(
                {
                    "source": "claude",
                    "session_id": session_id,
                    "session_path": str(jsonl_file),
                    "session_modified": session_modified,
                    "line_no": line_no,
                    "timestamp": entry.get("timestamp"),
                    "text": text,
                }
            )
    return messages


def extract_messages() -> list[dict[str, Any]]:
    messages = [
        *extract_codex_messages(),
        *extract_claude_messages(),
        *extract_copilot_messages(),
        *extract_copilot_cli_messages(),
    ]
    messages.sort(
        key=lambda message: (
            SOURCE_ORDER.index(message["source"]),
            message["session_modified"],
            _timestamp_sort_key(message.get("timestamp"), message["session_modified"]),
            message["line_no"],
        )
    )
    return messages


def _source_summary(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for source in SOURCE_ORDER:
        source_messages = [
            message for message in messages if message["source"] == source
        ]
        summary[source] = {
            "messages": len(source_messages),
            "sessions": len({message["session_path"] for message in source_messages}),
            "latest": max(
                (
                    message.get("timestamp") or message["session_modified"]
                    for message in source_messages
                ),
                default="",
            ),
        }
    return summary


def write_markdown(messages: list[dict[str, Any]], output: Path) -> None:
    summary = _source_summary(messages)
    lines = [
        "# User Messages - textToAudioTTS",
        "",
        f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Repo path: `{REPO_PATH}`",
        (
            "Scope: only sessions whose recorded workspace or cwd exactly matches this repo path. "
            "Includes Codex, Claude, Copilot VS Code, and Copilot CLI sessions."
        ),
        "Ordering: grouped by source, then session modification time, then message order.",
        f"Total messages: {len(messages)}",
        "",
        "## Source Summary",
        "",
    ]
    for source in SOURCE_ORDER:
        source_summary = summary[source]
        latest = source_summary["latest"] or "none"
        lines.append(
            f"- `{source}`: {source_summary['messages']} messages across "
            f"{source_summary['sessions']} sessions (latest: {latest})"
        )
    lines.extend(["", "---", ""])
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for message in messages:
        grouped[message["source"]][message["session_path"]].append(message)
    for source in SOURCE_ORDER:
        source_groups = grouped.get(source, {})
        lines.append(f"## Source: `{source}`")
        lines.append("")
        if not source_groups:
            lines.append("_No repo-scoped sessions found._")
            lines.extend(["", "---", ""])
            continue
        session_paths = sorted(
            source_groups,
            key=lambda session_path: (
                source_groups[session_path][0]["session_modified"],
                source_groups[session_path][0]["session_id"],
            ),
        )
        for session_path in session_paths:
            session_messages = source_groups[session_path]
            first_message = session_messages[0]
            lines.append(f"### Session: `{first_message['session_id']}`")
            lines.append(f"Path: `{session_path}`")
            lines.append(f"Modified: {first_message['session_modified']}")
            lines.append("")
            for message in session_messages:
                timestamp = message.get("timestamp") or ""
                timestamp_suffix = f" ({timestamp})" if timestamp else ""
                lines.append(
                    f"#### Message (line {message['line_no']}){timestamp_suffix}"
                )
                lines.append("")
                lines.append(message["text"])
                lines.extend(["", "---", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(messages)} user messages to {output}")
    for source in SOURCE_ORDER:
        source_summary = summary[source]
        print(
            f"{source}: {source_summary['messages']} messages across "
            f"{source_summary['sessions']} sessions"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract repo-scoped user messages from Codex, Copilot, and Claude transcripts"
    )
    parser.add_argument(
        "--output", default="docs/user_messages.md", help="Output markdown file"
    )
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = REPO_ROOT / output
    messages = extract_messages()
    write_markdown(messages, output)


if __name__ == "__main__":
    main()
