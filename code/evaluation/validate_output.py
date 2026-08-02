"""Whole-file submission gate for the final 110-row CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, OUTPUT_PATH  # noqa: E402
from data.schema import ACTIONS, MESSAGE_TYPES  # noqa: E402
from guards.decision import EVIDENCE_SEPARATOR, NO_EVIDENCE  # noqa: E402


EXPECTED_ROW_COUNT = 110
OUTPUT_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)


@dataclass(slots=True)
class RowFailure:
    line_number: int
    message_id: str
    row: dict[str | None, object]
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    output_path: Path
    row_count: int
    file_errors: tuple[str, ...]
    row_failures: tuple[RowFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.file_errors and not self.row_failures


def _id_set(path: Path, column: str) -> set[str]:
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or ()):
            raise ValueError(f"{path} is missing required column {column!r}")
        for row in reader:
            value = (row.get(column) or "").strip()
            if value:
                identifiers.add(value)
    return identifiers


def validate_output(
    output_path: Path,
    messages_path: Path,
    history_path: Path,
) -> ValidationReport:
    file_errors: list[str] = []
    failures: list[RowFailure] = []
    rows: list[RowFailure] = []
    try:
        expected_ids = _id_set(messages_path, "message_id")
        history_ids = _id_set(history_path, "message_id")
    except (OSError, ValueError, csv.Error) as error:
        return ValidationReport(output_path, 0, (str(error),), ())

    try:
        with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            actual_columns = tuple(reader.fieldnames or ())
            if actual_columns != OUTPUT_COLUMNS:
                file_errors.append(
                    "column order mismatch: "
                    f"expected={list(OUTPUT_COLUMNS)!r} actual={list(actual_columns)!r}"
                )
            try:
                for row in reader:
                    line_number = reader.line_num
                    message_id = (row.get("message_id") or "").strip()
                    failure = RowFailure(line_number, message_id or "<blank>", dict(row))
                    rows.append(failure)
                    if None in row:
                        failure.errors.append("row has more cells than the header")
                    for column, raw_value in row.items():
                        if column is None:
                            values = raw_value if isinstance(raw_value, list) else [raw_value]
                        else:
                            values = [raw_value]
                        if any(
                            "\n" in (value or "") or "\r" in (value or "")
                            for value in values
                            if isinstance(value, str) or value is None
                        ):
                            failure.errors.append(
                                f"embedded newline in cell {column or '<extra>'!r}"
                            )

                    action = (row.get("action") or "").strip()
                    message_type = (row.get("message_type") or "").strip()
                    reason = row.get("reason") or ""
                    confidence_raw = (row.get("confidence") or "").strip()
                    evidence_raw = (row.get("evidence_message_ids") or "").strip()
                    if not message_id:
                        failure.errors.append("blank message_id")
                    if action not in ACTIONS:
                        failure.errors.append(f"invalid action {action!r}")
                    if message_type not in MESSAGE_TYPES:
                        failure.errors.append(f"invalid message_type {message_type!r}")
                    if "\n" in reason or "\r" in reason:
                        failure.errors.append("reason must be a single line")
                    try:
                        confidence = float(confidence_raw)
                    except ValueError:
                        failure.errors.append(
                            f"confidence does not parse as a number: {confidence_raw!r}"
                        )
                    else:
                        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                            failure.errors.append(
                                f"confidence outside [0, 1]: {confidence_raw!r}"
                            )
                    if evidence_raw == NO_EVIDENCE:
                        evidence_ids: tuple[str, ...] = ()
                    elif not evidence_raw:
                        failure.errors.append(
                            "evidence_message_ids must be 'none' or historical ids"
                        )
                        evidence_ids = ()
                    else:
                        evidence_ids = tuple(
                            part.strip() for part in evidence_raw.split(EVIDENCE_SEPARATOR)
                        )
                        unresolved = sorted(
                            {
                                identifier
                                for identifier in evidence_ids
                                if not identifier or identifier not in history_ids
                            }
                        )
                        if unresolved:
                            failure.errors.append(
                                f"unresolved evidence ids: {unresolved!r}"
                            )
            except csv.Error as error:
                file_errors.append(f"malformed CSV near physical line {reader.line_num}: {error}")
    except OSError as error:
        return ValidationReport(output_path, 0, (str(error),), ())

    if len(rows) != EXPECTED_ROW_COUNT:
        file_errors.append(
            f"expected exactly {EXPECTED_ROW_COUNT} rows, found {len(rows)}"
        )

    by_id: dict[str, list[RowFailure]] = {}
    for row in rows:
        if row.message_id != "<blank>":
            by_id.setdefault(row.message_id, []).append(row)
    for message_id, duplicates in sorted(by_id.items()):
        if len(duplicates) > 1:
            lines = [row.line_number for row in duplicates]
            for duplicate in duplicates:
                duplicate.errors.append(
                    f"duplicate message_id {message_id!r} on lines {lines}"
                )

    actual_ids = set(by_id)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids:
        file_errors.append(f"message_id set is missing: {missing_ids!r}")
    if extra_ids:
        file_errors.append(f"message_id set has extras: {extra_ids!r}")
        for message_id in extra_ids:
            for row in by_id[message_id]:
                row.errors.append("message_id does not exist in messages.csv")

    failures.extend(row for row in rows if row.errors)
    return ValidationReport(
        output_path=output_path,
        row_count=len(rows),
        file_errors=tuple(file_errors),
        row_failures=tuple(failures),
    )


def print_validation_report(report: ValidationReport) -> None:
    if report.ok:
        print(
            f"SUBMISSION GATE PASS: {report.output_path} has {report.row_count} valid rows"
        )
        return
    print(f"SUBMISSION GATE FAIL: {report.output_path}")
    for error in report.file_errors:
        print(f"FILE: {error}")
    for failure in report.row_failures:
        rendered = json.dumps(failure.row, ensure_ascii=False, default=str)
        print(
            f"ROW line={failure.line_number} message_id={failure.message_id!r}: "
            f"{'; '.join(failure.errors)}"
        )
        print(f"  {rendered}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the complete submission CSV.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--messages", type=Path, default=DATASET_DIR / "messages.csv")
    parser.add_argument(
        "--history", type=Path, default=DATASET_DIR / "message_history.csv"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_output(args.output, args.messages, args.history)
    print_validation_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
