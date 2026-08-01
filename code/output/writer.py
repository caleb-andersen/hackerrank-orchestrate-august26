"""Incremental CSV output that preserves the template's message order."""

import csv
import os
import threading
from pathlib import Path
from typing import IO, Mapping, Self


class _OrderedWriter:
    def __init__(self, path: str | Path, ordered_message_ids: list[str]) -> None:
        if len(set(ordered_message_ids)) != len(ordered_message_ids):
            raise ValueError("ordered_message_ids contains duplicates")
        self._ordered_message_ids = list(ordered_message_ids)
        self._allowed_message_ids = set(ordered_message_ids)
        self._pending: dict[str, dict[str, str]] = {}
        self._written: set[str] = set()
        self._next_index = 0
        self._lock = threading.Lock()
        self._closed = False

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = output_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=(
                "message_id",
                "action",
                "message_type",
                "reason",
                "confidence",
                "evidence_message_ids",
            ),
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        self._writer.writeheader()
        self._sync()

    @staticmethod
    def _single_line(value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).splitlines())

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def append_row(self, row_dict: Mapping[str, object]) -> None:
        with self._lock:
            if self._closed:
                raise ValueError("Cannot append to a closed output writer")
            message_id = self._single_line(row_dict.get("message_id"))
            if not message_id:
                raise ValueError("Output row requires a non-empty message_id")
            if message_id not in self._allowed_message_ids:
                raise ValueError(f"Output row has an unexpected message_id: {message_id!r}")
            if message_id in self._written or message_id in self._pending:
                raise ValueError(f"Output row is duplicated: {message_id!r}")

            normalized = {
                column: self._single_line(row_dict.get(column))
                for column in (
                    "message_id",
                    "action",
                    "message_type",
                    "reason",
                    "confidence",
                    "evidence_message_ids",
                )
            }
            normalized["message_id"] = message_id
            self._pending[message_id] = normalized
            self._write_ready_rows()

    def _write_ready_rows(self) -> None:
        while self._next_index < len(self._ordered_message_ids):
            expected_id = self._ordered_message_ids[self._next_index]
            row = self._pending.get(expected_id)
            if row is None:
                break
            self._writer.writerow(row)
            self._sync()
            del self._pending[expected_id]
            self._written.add(expected_id)
            self._next_index += 1

    def close(self) -> None:
        global _active_writer
        with self._lock:
            if self._closed:
                return
            self._sync()
            self._handle.close()
            self._closed = True
            if _active_writer is self:
                _active_writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


_active_writer: _OrderedWriter | None = None


def open_writer(path: str | Path, ordered_message_ids: list[str]) -> _OrderedWriter:
    """Open the active ordered writer and persist its header immediately."""
    global _active_writer
    if _active_writer is not None:
        raise RuntimeError("An output writer is already open")
    _active_writer = _OrderedWriter(path, ordered_message_ids)
    return _active_writer


def append_row(row_dict: Mapping[str, object]) -> None:
    """Append a completed row, buffering it until earlier ordered rows exist."""
    if _active_writer is None:
        raise RuntimeError("open_writer() must be called before append_row()")
    _active_writer.append_row(row_dict)
