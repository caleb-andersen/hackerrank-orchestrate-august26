"""Measure the deterministic routing signal available before any model is used.

This rule-only baseline is for calibration, not accuracy: it shows how much of
the sample label signal the Dossier can explain deterministically and remains in
the repository as the floor the agent has to beat. It makes no model calls.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config  # noqa: E402
from context import features as dossier_features  # noqa: E402
from context.features import Dossier  # noqa: E402
from context.index import build_feature_index  # noqa: E402
from data.loader import Dataset, load_dataset  # noqa: E402
from data.schema import ACTIONS, Message  # noqa: E402


PROMOTION_PATTERN = re.compile(
    r"\b(?:sale|discount|promo(?:tion)?|offer|deal|coupon|clearance|"
    r"buy\s+one|get\s+one|free\s+(?:gift|shipping)|limited\s+time)\b|"
    r"\b\d{1,2}%\s*off\b"
)
DIRECT_MENTION_PATTERN = re.compile(r"(?<![\w@])@[a-z0-9_.-]+\b")
DEADLINE_PATTERN = re.compile(
    r"\b(?:deadline|due\s+(?:today|tonight|tomorrow|by)|expires?|expiry|"
    r"last\s+(?:day|chance)|final\s+(?:notice|reminder)|urgent|asap|"
    r"immediately)\b"
)

# Ordered from specific to general. The first matching category wins.
MESSAGE_TYPE_KEYWORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "payment",
        re.compile(
            r"\b(?:pay(?:ment)?|invoice|bill|amount\s+due|transfer|fee|balance)\b"
        ),
    ),
    (
        "promotion",
        PROMOTION_PATTERN,
    ),
    (
        "event",
        re.compile(
            r"\b(?:event|meeting|appointment|party|webinar|workshop|conference|"
            r"venue|rsvp)\b"
        ),
    ),
    (
        "business_update",
        re.compile(
            r"\b(?:order|delivery|delivered|shipping|shipment|booking|account|"
            r"status|tracking|service\s+update)\b"
        ),
    ),
    (
        "greeting",
        re.compile(
            r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|"
            r"happy\s+(?:birthday|anniversary))\b"
        ),
    ),
    (
        "forward",
        re.compile(r"\b(?:forwarded|forward\s+this|share\s+this)\b"),
    ),
    (
        "urgent",
        DEADLINE_PATTERN,
    ),
)

SWEEP_VALUES: dict[str, tuple[int | float, ...]] = {
    "BRAND_MIN_AGE_DAYS": (180, 270, 365, 450, 540),
    "BRAND_MAX_REPORTS": (10, 20, 29, 40, 50),
    "DISMISS_MUTE_THRESHOLD": (0.3, 0.4, 0.5, 0.6, 0.7),
    "MIN_PEER_HISTORY": (1, 2, 3, 4, 5),
}


@dataclass(frozen=True, slots=True)
class Thresholds:
    brand_min_age_days: int = config.BRAND_MIN_AGE_DAYS
    brand_max_reports: int = config.BRAND_MAX_REPORTS
    dismiss_mute_threshold: float = config.DISMISS_MUTE_THRESHOLD
    min_peer_history: int = config.MIN_PEER_HISTORY


@dataclass(frozen=True, slots=True)
class LabelledMessage:
    message: Message
    action: str
    message_type: str


@dataclass(frozen=True, slots=True)
class Prediction:
    action: str
    message_type: str
    rule: str


@dataclass(frozen=True, slots=True)
class Scores:
    total: int
    action_correct: int
    message_type_correct: int
    joint_correct: int
    confusion: dict[tuple[str, str], int]
    predicted_actions: Counter[str]
    gold_actions: Counter[str]

    @property
    def action_accuracy(self) -> float:
        return self.action_correct / self.total

    @property
    def message_type_accuracy(self) -> float:
        return self.message_type_correct / self.total

    @property
    def joint_accuracy(self) -> float:
        return self.joint_correct / self.total


def _required(row: dict[str, str | None], key: str, path: Path) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"{path} has a blank required value for {key}")
    return value.strip()


def _optional(row: dict[str, str | None], key: str) -> str | None:
    value = row.get(key)
    if value is None or not value.strip():
        return None
    return value.strip()


def _parse_datetime(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)


def load_labelled_samples(path: Path) -> tuple[LabelledMessage, ...]:
    """Load labels only inside the evaluation package."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        required_headers = {
            "message_id",
            "user_id",
            "conversation_type",
            "created_at",
            "message_text",
            "forwarded_count",
            "action",
            "message_type",
        }
        missing = required_headers.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        samples: list[LabelledMessage] = []
        for row in reader:
            message = Message(
                message_id=_required(row, "message_id", path),
                user_id=_required(row, "user_id", path),
                conversation_type=_required(row, "conversation_type", path),
                group_id=_optional(row, "group_id"),
                business_id=_optional(row, "business_id"),
                sender_user_id=_optional(row, "sender_user_id"),
                created_at=_parse_datetime(_required(row, "created_at", path)),
                message_text=row.get("message_text") or "",
                media_type=_optional(row, "media_type"),
                media_id=_optional(row, "media_id"),
                forwarded_count=int(_required(row, "forwarded_count", path)),
            )
            samples.append(
                LabelledMessage(
                    message=message,
                    action=_required(row, "action", path).casefold(),
                    message_type=_required(row, "message_type", path).casefold(),
                )
            )
    if not samples:
        raise ValueError(f"{path} contains no sample rows")
    return tuple(samples)


def _is_promotional(dossier: Dossier) -> bool:
    text = dossier.content_signals.raw_text.casefold()
    return PROMOTION_PATTERN.search(text) is not None


def _has_direct_mention(dossier: Dossier) -> bool:
    # The Dossier's normalized text intentionally strips punctuation, including @.
    text = dossier.content_signals.raw_text.casefold()
    return DIRECT_MENTION_PATTERN.search(text) is not None


def _has_deadline_marker(dossier: Dossier) -> bool:
    return DEADLINE_PATTERN.search(dossier.content_signals.normalised_text) is not None


def _keyword_message_type(dossier: Dossier) -> str:
    text = dossier.content_signals.raw_text.casefold()
    for message_type, pattern in MESSAGE_TYPE_KEYWORDS:
        if pattern.search(text) is not None:
            return message_type
    if dossier.conversation_type == "personal":
        return "personal"
    return "unknown"


def route(dossier: Dossier, thresholds: Thresholds) -> Prediction:
    """Apply the deterministic routing rules in their required priority order."""
    content = dossier.content_signals
    if content.injection_match is not None:
        return Prediction("mute", "scam", "injection_match")
    if content.credential_request is not None:
        return Prediction("mute", "scam", "credential_request")

    integrity = dossier.sender_identity.brand_integrity
    if integrity is not None and integrity.verdict == "impersonation":
        return Prediction("mute", "scam", "brand_impersonation")

    business = dossier.relationship.business_relationship
    if (
        business is not None
        and business.promotions_opted_out_at is not None
        and _is_promotional(dossier)
    ):
        return Prediction("mute", "promotion", "promotion_opt_out")

    engagement = dossier.relationship.peer_engagement
    if (
        engagement.dismiss_rate is not None
        and engagement.dismiss_rate >= thresholds.dismiss_mute_threshold
        and engagement.n >= thresholds.min_peer_history
    ):
        return Prediction("mute", _keyword_message_type(dossier), "peer_dismiss_rate")

    if _has_direct_mention(dossier) or _has_deadline_marker(dossier):
        return Prediction("notify", _keyword_message_type(dossier), "attention_marker")

    return Prediction("digest", _keyword_message_type(dossier), "default")


@contextmanager
def _brand_thresholds(thresholds: Thresholds) -> Iterator[None]:
    """Temporarily apply brand thresholds used while Dossiers are built."""
    previous_age = dossier_features.BRAND_MIN_AGE_DAYS
    previous_reports = dossier_features.BRAND_MAX_REPORTS
    dossier_features.BRAND_MIN_AGE_DAYS = thresholds.brand_min_age_days
    dossier_features.BRAND_MAX_REPORTS = thresholds.brand_max_reports
    try:
        yield
    finally:
        dossier_features.BRAND_MIN_AGE_DAYS = previous_age
        dossier_features.BRAND_MAX_REPORTS = previous_reports


def evaluate(
    dataset: Dataset,
    samples: Sequence[LabelledMessage],
    thresholds: Thresholds,
) -> Scores:
    index = build_feature_index(dataset)
    confusion = {(gold, predicted): 0 for gold in ACTIONS for predicted in ACTIONS}
    predicted_actions: Counter[str] = Counter()
    gold_actions: Counter[str] = Counter()
    action_correct = 0
    message_type_correct = 0
    joint_correct = 0

    with _brand_thresholds(thresholds):
        for sample in samples:
            dossier = dossier_features.build_dossier(dataset, index, sample.message)
            prediction = route(dossier, thresholds)
            if sample.action not in ACTIONS:
                raise ValueError(f"Unknown gold action: {sample.action}")
            if prediction.action not in ACTIONS:
                raise ValueError(f"Unknown predicted action: {prediction.action}")
            action_match = prediction.action == sample.action
            type_match = prediction.message_type == sample.message_type
            action_correct += action_match
            message_type_correct += type_match
            joint_correct += action_match and type_match
            confusion[(sample.action, prediction.action)] += 1
            predicted_actions[prediction.action] += 1
            gold_actions[sample.action] += 1

    return Scores(
        total=len(samples),
        action_correct=action_correct,
        message_type_correct=message_type_correct,
        joint_correct=joint_correct,
        confusion=confusion,
        predicted_actions=predicted_actions,
        gold_actions=gold_actions,
    )


def _format_accuracy(correct: int, total: int) -> str:
    return f"{correct / total:.3f} ({correct}/{total})"


def print_summary(scores: Scores) -> None:
    print(f"Action accuracy:       {_format_accuracy(scores.action_correct, scores.total)}")
    print(
        "Message-type accuracy: "
        f"{_format_accuracy(scores.message_type_correct, scores.total)}"
    )
    print()
    print("Action confusion matrix (rows=gold, columns=predicted)")
    print(f"{'gold \\ pred':<14}" + "".join(f"{action:>9}" for action in ACTIONS))
    for gold in ACTIONS:
        counts = "".join(
            f"{scores.confusion[(gold, predicted)]:>9}" for predicted in ACTIONS
        )
        print(f"{gold:<14}{counts}")

    print()
    print("Action distributions")
    print(f"{'action':<10}{'predicted':>12}{'gold':>12}")
    for action in ACTIONS:
        print(
            f"{action:<10}{scores.predicted_actions[action]:>12}"
            f"{scores.gold_actions[action]:>12}"
        )


def _with_sweep_value(
    thresholds: Thresholds,
    name: str,
    value: int | float,
) -> Thresholds:
    fields = {
        "BRAND_MIN_AGE_DAYS": "brand_min_age_days",
        "BRAND_MAX_REPORTS": "brand_max_reports",
        "DISMISS_MUTE_THRESHOLD": "dismiss_mute_threshold",
        "MIN_PEER_HISTORY": "min_peer_history",
    }
    return replace(thresholds, **{fields[name]: value})


def print_sweeps(
    dataset: Dataset,
    samples: Sequence[LabelledMessage],
    defaults: Thresholds,
) -> None:
    print()
    print("One-at-a-time threshold sweeps")
    print("(* = maximum action accuracy; all other thresholds held at defaults)")
    for name, values in SWEEP_VALUES.items():
        results = [
            (
                value,
                evaluate(
                    dataset,
                    samples,
                    _with_sweep_value(defaults, name, value),
                ),
            )
            for value in values
        ]
        best_action_correct = max(scores.action_correct for _value, scores in results)
        best_values = [
            value
            for value, scores in results
            if scores.action_correct == best_action_correct
        ]
        print()
        print(name)
        print(f"{'value':>10}  {'action':>9}  {'type':>9}  {'joint':>9}  best")
        for value, scores in results:
            marker = "*" if scores.action_correct == best_action_correct else ""
            print(
                f"{value:>10}  {scores.action_accuracy:>9.3f}  "
                f"{scores.message_type_accuracy:>9.3f}  "
                f"{scores.joint_accuracy:>9.3f}  {marker}"
            )
        joined = ", ".join(str(value) for value in best_values)
        print(
            f"Best action accuracy: {best_action_correct / len(samples):.3f} "
            f"at {joined}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="Participant dataset directory (default: repository dataset/)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    dataset = load_dataset(dataset_dir)
    samples = load_labelled_samples(dataset_dir / "sample_messages.csv")
    defaults = Thresholds()

    print("Rule-only Dossier baseline (no model calls)")
    print(
        "Thresholds: "
        f"BRAND_MIN_AGE_DAYS={defaults.brand_min_age_days}, "
        f"BRAND_MAX_REPORTS={defaults.brand_max_reports}, "
        f"DISMISS_MUTE_THRESHOLD={defaults.dismiss_mute_threshold}, "
        f"MIN_PEER_HISTORY={defaults.min_peer_history}"
    )
    print()
    print_summary(evaluate(dataset, samples, defaults))
    print_sweeps(dataset, samples, defaults)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
