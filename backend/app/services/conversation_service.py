"""Per-user chat storage: one JSON file per conversation under data/conversations/<user>/.

Separate small files (rather than one shared file or database) let OneDrive sync chats
between devices without one device's save overwriting another's unrelated chats.
"""
import json
import os
import re
from pathlib import Path

from app.config import CONVERSATIONS_DIR

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ConversationError(ValueError):
    """Raised for an invalid conversation id or body."""


def _user_dir(username: str) -> Path:
    # Usernames are case-insensitive (see users table); keep the folder name filesystem-safe.
    safe = re.sub(r"[^a-z0-9_-]", "_", username.lower())
    return CONVERSATIONS_DIR / safe


def _path(username: str, conversation_id: str) -> Path:
    if not _ID_RE.match(conversation_id):
        raise ConversationError("Invalid conversation id.")
    return _user_dir(username) / f"{conversation_id}.json"


def list_conversations(username: str) -> list[dict]:
    """All of a user's conversations, most recently updated first. Unreadable files are skipped."""
    folder = _user_dir(username)
    if not folder.is_dir():
        return []
    out = []
    for p in folder.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # e.g. a OneDrive placeholder mid-download or a half-synced file
            continue
        if isinstance(data, dict) and data.get("id") == p.stem:
            out.append(data)
    out.sort(key=lambda c: c.get("updatedAt") or 0, reverse=True)
    return out


def save_conversation(username: str, conversation_id: str, data: dict) -> dict:
    """Create or replace a conversation. Written atomically so a sync never sees half a file."""
    path = _path(username, conversation_id)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ConversationError("Conversation must be an object with a messages list.")
    data = {**data, "id": conversation_id}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return data


def delete_conversation(username: str, conversation_id: str) -> None:
    _path(username, conversation_id).unlink(missing_ok=True)
