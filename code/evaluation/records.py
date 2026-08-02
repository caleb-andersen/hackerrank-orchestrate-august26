"""Strict CSV records shared only by offline evaluation code."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from data.schema import Message
from guards.decision import EVIDENCE_SEPARATOR, NO_EVIDENCE


@dataclass(frozen=True, slots=True)
class GoldSample:
    message: Message
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class Prediction:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: frozenset[str]


def parse_evidence(value: str) -> frozenset[str]:
    cleaned = value.strip()
    if cleaned == NO_EVIDENCE:
        return frozenset()
    if not cleaned:
        raise ValueError("evidence_message_ids must be 'none' or semicolon-separated ids")
    identifiers = tuple(part.strip() for part in cleaned.split(EVIDENCE_SEPARATOR))
    if any(not identifier or identifier == NO_EVIDENCE for identifier in identifiers):
        raise ValueError(f"invalid evidence_message_ids value: {value!r}")
    return frozenset(identifiers)


def _required(row: dict[str, str | None], field: str, path: Path, line: int) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f"{path} line {line}: blank required field {field!r}")
    return value.strip()


def _optional(row: dict[str, str | None], field: str) -> str | None:
    value = row.get(field)
    return None if value is None or not value.strip() else value.strip()


def _datetime(value: str, path: Path, line: int) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{path} line {line}: invalid created_at {value!r}") from error


def _confidence(value: str, path: Path, line: int) -> float:
    try:
        confidence = float(value)
    except ValueError as error:
        raise ValueError(f"{path} line {line}: invalid confidence {value!r}") from error
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{path} line {line}: confidence outside [0, 1]: {value!r}")
    return confidence


def load_gold_samples(path: Path) -> tuple[GoldSample, ...]:
    """Load labelled samples here, never through the production data package."""
    required_headers = (
        "message_id",
        "user_id",
        "conversation_type",
        "group_id",
        "business_id",
        "sender_user_id",
        "created_at",
        "message_text",
        "media_type",
        "media_id",
        "forwarded_count",
        "action",
        "message_type",
        "reason",
        "confidence",
        "evidence_message_ids",
    )
    samples: list[GoldSample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in required_headers if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            line = reader.line_num
            if None in row:
                raise ValueError(f"{path} line {line}: more cells than columns")
            message_id = _required(row, "message_id", path, line)
            if message_id in seen:
                raise ValueError(f"{path} line {line}: duplicate message_id {message_id!r}")
            seen.add(message_id)
            try:
                forwarded_count = int(_required(row, "forwarded_count", path, line))
            except ValueError as error:
                raise ValueError(
                    f"{path} line {line}: forwarded_count must be an integer"
                ) from error
            message = Message(
                message_id=message_id,
                user_id=_required(row, "user_id", path, line),
                conversation_type=_required(row, "conversation_type", path, line),
                group_id=_optional(row, "group_id"),
                business_id=_optional(row, "business_id"),
                sender_user_id=_optional(row, "sender_user_id"),
                created_at=_datetime(_required(row, "created_at", path, line), path, line),
                message_text=row.get("message_text") or "",
                media_type=_optional(row, "media_type"),
                media_id=_optional(row, "media_id"),
                forwarded_count=forwarded_count,
            )
            samples.append(
                GoldSample(
                    message=message,
                    action=_required(row, "action", path, line).casefold(),
                    message_type=_required(row, "message_type", path, line).casefold(),
                    reason=_required(row, "reason", path, line),
                    confidence=_confidence(
                        _required(row, "confidence", path, line), path, line
                    ),
                    evidence_message_ids=parse_evidence(
                        _required(row, "evidence_message_ids", path, line)
                    ),
                )
            )
    if not samples:
        raise ValueError(f"{path} contains no labelled rows")
    return tuple(samples)


def load_predictions(path: Path) -> dict[str, Prediction]:
    required_headers = (
        "message_id",
        "action",
        "message_type",
        "reason",
        "confidence",
        "evidence_message_ids",
    )
    predictions: dict[str, Prediction] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in required_headers if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            line = reader.line_num
            if None in row:
                raise ValueError(f"{path} line {line}: more cells than columns")
            message_id = _required(row, "message_id", path, line)
            if message_id in predictions:
                raise ValueError(f"{path} line {line}: duplicate message_id {message_id!r}")
            predictions[message_id] = Prediction(
                message_id=message_id,
                action=_required(row, "action", path, line).casefold(),
                message_type=_required(row, "message_type", path, line).casefold(),
                reason=_required(row, "reason", path, line),
                confidence=_confidence(
                    _required(row, "confidence", path, line), path, line
                ),
                evidence_message_ids=parse_evidence(
                    _required(row, "evidence_message_ids", path, line)
                ),
            )
    return predictions
