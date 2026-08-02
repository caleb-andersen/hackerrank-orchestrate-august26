"""Near-duplicate action consistency audit over the full prediction set."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass, is_dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, OUTPUT_PATH  # noqa: E402
from context.features import Dossier, build_dossier  # noqa: E402
from context.index import build_feature_index  # noqa: E402
from context.text import jaccard, normalise_text, trigrams  # noqa: E402
from data.loader import load_dataset  # noqa: E402
from data.schema import Message  # noqa: E402


DEFAULT_THRESHOLD = 0.85
_EXCLUDED_FEATURE_PARTS = frozenset(
    {
        "message_id",
        "user_id",
        "peer_id",
        "group_id",
        "display_name",
        "brand_name",
        "group_name",
        "basis_note",
        "created_at",
        "last_activity_at",
        "promotions_opted_out_at",
        "last_reply_at",
    }
)


@dataclass(frozen=True, slots=True)
class Divergence:
    cluster_number: int
    left_id: str
    right_id: str
    left_action: str
    right_action: str
    similarity: float
    feature_differences: tuple[tuple[str, object, object], ...]

    @property
    def is_bug(self) -> bool:
        return not self.feature_differences


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    message_count: int
    cluster_count: int
    divergent_pairs: tuple[Divergence, ...]
    missing_prediction_ids: tuple[str, ...]

    @property
    def bug_count(self) -> int:
        return sum(divergence.is_bug for divergence in self.divergent_pairs)


def _load_actions(path: Path) -> dict[str, str]:
    actions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in ("message_id", "action")
            if column not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row in reader:
            message_id = (row.get("message_id") or "").strip()
            if not message_id:
                continue
            if message_id in actions:
                raise ValueError(f"{path} contains duplicate message_id {message_id!r}")
            actions[message_id] = (row.get("action") or "").strip()
    return actions


def _flatten(prefix: str, value: object, result: dict[str, object]) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        for key in sorted(value):
            if str(key) in _EXCLUDED_FEATURE_PARTS:
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child, value[key], result)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        result[prefix] = tuple(value)
        return
    result[prefix] = value


def personalisation_features(dossier: Dossier) -> dict[str, object]:
    """Return decision-relevant context while excluding text, labels, and opaque ids."""
    source = {
        "conversation_type": dossier.conversation_type,
        "sender": dossier.sender_identity,
        "relationship": dossier.relationship,
        "timing": {
            "dnd_window_raw": dossier.timing.dnd_window_raw,
            "in_dnd": dossier.timing.in_dnd,
            "minutes_until_dnd_ends": dossier.timing.minutes_until_dnd_ends,
        },
    }
    flattened: dict[str, object] = {}
    _flatten("", source, flattened)
    return flattened


def _clusters(messages: Sequence[Message], threshold: float) -> tuple[tuple[int, ...], ...]:
    grams = [trigrams(normalise_text(message.message_text)) for message in messages]
    parent = list(range(len(messages)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in combinations(range(len(messages)), 2):
        similarity = jaccard(grams[left], grams[right])
        if similarity is not None and similarity > threshold:
            union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(len(messages)):
        grouped.setdefault(find(index), []).append(index)
    clusters = [tuple(indices) for indices in grouped.values() if len(indices) > 1]
    return tuple(sorted(clusters, key=lambda values: messages[values[0]].message_id))


def audit_messages(
    messages: Sequence[Message],
    actions: Mapping[str, str],
    features: Mapping[str, Mapping[str, object]],
    threshold: float = DEFAULT_THRESHOLD,
) -> ConsistencyReport:
    clusters = _clusters(messages, threshold)
    missing = tuple(
        message.message_id for message in messages if message.message_id not in actions
    )
    grams = {
        message.message_id: trigrams(normalise_text(message.message_text))
        for message in messages
    }
    divergences: list[Divergence] = []
    for cluster_number, indices in enumerate(clusters, start=1):
        for left_index, right_index in combinations(indices, 2):
            left = messages[left_index]
            right = messages[right_index]
            left_action = actions.get(left.message_id)
            right_action = actions.get(right.message_id)
            if left_action is None or right_action is None or left_action == right_action:
                continue
            left_features = features[left.message_id]
            right_features = features[right.message_id]
            differing_keys = sorted(set(left_features) | set(right_features))
            differences = tuple(
                (key, left_features.get(key), right_features.get(key))
                for key in differing_keys
                if left_features.get(key) != right_features.get(key)
            )
            similarity = jaccard(grams[left.message_id], grams[right.message_id])
            divergences.append(
                Divergence(
                    cluster_number=cluster_number,
                    left_id=left.message_id,
                    right_id=right.message_id,
                    left_action=left_action,
                    right_action=right_action,
                    similarity=0.0 if similarity is None else similarity,
                    feature_differences=differences,
                )
            )
    return ConsistencyReport(
        message_count=len(messages),
        cluster_count=len(clusters),
        divergent_pairs=tuple(divergences),
        missing_prediction_ids=missing,
    )


def audit_full_predictions(
    dataset_dir: Path,
    predictions_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> ConsistencyReport:
    dataset = load_dataset(dataset_dir)
    index = build_feature_index(dataset)
    actions = _load_actions(predictions_path)
    features = {
        message.message_id: personalisation_features(
            build_dossier(dataset, index, message)
        )
        for message in dataset.messages
    }
    return audit_messages(dataset.messages, actions, features, threshold)


def print_consistency_report(report: ConsistencyReport) -> None:
    print(
        "NEAR-DUPLICATE CONSISTENCY: "
        f"{report.message_count} full-set rows, {report.cluster_count} clusters, "
        f"{len(report.divergent_pairs)} divergent action pairs, {report.bug_count} bugs"
    )
    if report.missing_prediction_ids:
        print("MISSING PREDICTIONS: " + ", ".join(report.missing_prediction_ids))
    for divergence in report.divergent_pairs:
        status = "BUG" if divergence.is_bug else "CORRECT DIVERGENCE"
        print(
            f"{status}: cluster={divergence.cluster_number} "
            f"{divergence.left_id}({divergence.left_action}) vs "
            f"{divergence.right_id}({divergence.right_action}) "
            f"trigram_jaccard={divergence.similarity:.4f}"
        )
        if divergence.feature_differences:
            print("  personalisation features that differ:")
            for name, left, right in divergence.feature_differences:
                print(f"    {name}: {left!r} != {right!r}")
        else:
            print("  BUG: no personalisation feature differs")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit near-duplicate actions across the full prediction set."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--predictions", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_full_predictions(args.dataset, args.predictions, args.threshold)
    except (OSError, ValueError, csv.Error) as error:
        print(f"CONSISTENCY AUDIT FAIL: {error}")
        return 1
    print_consistency_report(report)
    return 1 if report.bug_count or report.missing_prediction_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
