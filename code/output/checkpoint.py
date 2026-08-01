"""Crash-resilient append-only JSONL checkpoints."""

import json
import os
import threading
from pathlib import Path
from typing import Mapping

from config import CACHE_DIR


_append_lock = threading.Lock()


def _checkpoint_path() -> Path:
    return CACHE_DIR / "checkpoint.jsonl"


def save(message_id: str, fingerprint: str, row: Mapping[str, object]) -> None:
    """Atomically append one complete checkpoint record and force it to disk."""
    if not message_id:
        raise ValueError("message_id must not be empty")
    if not fingerprint:
        raise ValueError("fingerprint must not be empty")

    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"message_id": message_id, "fingerprint": fingerprint, "row": dict(row)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)

    with _append_lock:
        descriptor = os.open(path, flags, 0o600)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(f"Incomplete checkpoint append: wrote {written} of {len(payload)} bytes")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def load_completed() -> dict[str, tuple[str, dict[str, object]]]:
    """Return the newest completed checkpoint for every message ID."""
    path = _checkpoint_path()
    if not path.exists():
        return {}

    completed: dict[str, tuple[str, dict[str, object]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if not line.endswith("\n"):
                    break
                raise ValueError(f"Malformed checkpoint JSON on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Checkpoint line {line_number} is not an object")
            message_id = record.get("message_id")
            fingerprint = record.get("fingerprint")
            row = record.get("row")
            if not isinstance(message_id, str) or not message_id:
                raise ValueError(f"Checkpoint line {line_number} has an invalid message_id")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError(f"Checkpoint line {line_number} has an invalid fingerprint")
            if not isinstance(row, dict) or not all(isinstance(key, str) for key in row):
                raise ValueError(f"Checkpoint line {line_number} has an invalid row")
            completed[message_id] = (fingerprint, row)
    return completed
