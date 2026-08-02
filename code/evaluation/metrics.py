"""Multi-column evaluation metrics for labelled sample predictions."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.schema import ACTIONS, MESSAGE_TYPES  # noqa: E402
from evaluation.records import GoldSample, Prediction  # noqa: E402


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float | None
    mean_gold: float | None
    mean_absolute_error: float | None
    action_accuracy: float | None


@dataclass(frozen=True, slots=True)
class MetricReport:
    total: int
    action_accuracy: float
    type_accuracy: float
    joint_accuracy: float
    action_confusion: Mapping[tuple[str, str], int]
    type_confusion: Mapping[tuple[str, str], int]
    catastrophic_action_count: int
    catastrophic_scam_count: int
    miss_count: int
    evidence_precision: float
    evidence_recall: float
    evidence_f1: float
    calibration_error: float
    reliability: tuple[ReliabilityBin, ...]
    gold_actions: Mapping[str, int]
    predicted_actions: Mapping[str, int]
    gold_types: Mapping[str, int]
    predicted_types: Mapping[str, int]
    missing_prediction_ids: tuple[str, ...]
    extra_prediction_ids: tuple[str, ...]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _evidence_scores(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
    if true_positive == false_positive == false_negative == 0:
        return (1.0, 1.0, 1.0)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return (precision, recall, f1)


def _reliability_bins(
    samples: Sequence[tuple[float, float, bool]],
) -> tuple[ReliabilityBin, ...]:
    buckets: list[list[tuple[float, float, bool]]] = [[] for _ in range(20)]
    for predicted, gold, action_correct in samples:
        index = min(int((predicted + 1e-12) * 20), 19)
        buckets[index].append((predicted, gold, action_correct))

    result: list[ReliabilityBin] = []
    for index, bucket in enumerate(buckets):
        lower = index / 20
        upper = (index + 1) / 20
        if not bucket:
            result.append(ReliabilityBin(lower, upper, 0, None, None, None, None))
            continue
        count = len(bucket)
        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_predicted=sum(item[0] for item in bucket) / count,
                mean_gold=sum(item[1] for item in bucket) / count,
                mean_absolute_error=sum(abs(item[0] - item[1]) for item in bucket)
                / count,
                action_accuracy=sum(item[2] for item in bucket) / count,
            )
        )
    return tuple(result)


def evaluate_metrics(
    gold: Sequence[GoldSample], predictions: Mapping[str, Prediction]
) -> MetricReport:
    total = len(gold)
    gold_ids = {sample.message.message_id for sample in gold}
    missing = tuple(
        sample.message.message_id
        for sample in gold
        if sample.message.message_id not in predictions
    )
    extra = tuple(sorted(set(predictions) - gold_ids))

    action_correct = 0
    type_correct = 0
    joint_correct = 0
    evidence_tp = 0
    evidence_fp = 0
    evidence_fn = 0
    confidence_errors: list[float] = []
    reliability_samples: list[tuple[float, float, bool]] = []
    action_confusion: Counter[tuple[str, str]] = Counter()
    type_confusion: Counter[tuple[str, str]] = Counter()
    gold_actions: Counter[str] = Counter()
    predicted_actions: Counter[str] = Counter()
    gold_types: Counter[str] = Counter()
    predicted_types: Counter[str] = Counter()
    catastrophic_action_count = 0
    catastrophic_scam_count = 0
    miss_count = 0

    for sample in gold:
        gold_actions[sample.action] += 1
        gold_types[sample.message_type] += 1
        prediction = predictions.get(sample.message.message_id)
        if prediction is None:
            evidence_fn += len(sample.evidence_message_ids)
            confidence_errors.append(1.0)
            continue

        predicted_actions[prediction.action] += 1
        predicted_types[prediction.message_type] += 1
        action_ok = sample.action == prediction.action
        type_ok = sample.message_type == prediction.message_type
        action_correct += int(action_ok)
        type_correct += int(type_ok)
        joint_correct += int(action_ok and type_ok)
        action_confusion[(sample.action, prediction.action)] += 1
        type_confusion[(sample.message_type, prediction.message_type)] += 1

        if sample.action == "mute" and prediction.action == "notify":
            catastrophic_action_count += 1
            if sample.message_type == "scam":
                catastrophic_scam_count += 1
        if sample.action == "notify" and prediction.action == "mute":
            miss_count += 1

        expected = sample.evidence_message_ids
        produced = prediction.evidence_message_ids
        evidence_tp += len(expected & produced)
        evidence_fp += len(produced - expected)
        evidence_fn += len(expected - produced)
        confidence_errors.append(abs(prediction.confidence - sample.confidence))
        reliability_samples.append((prediction.confidence, sample.confidence, action_ok))

    evidence_precision, evidence_recall, evidence_f1 = _evidence_scores(
        evidence_tp, evidence_fp, evidence_fn
    )
    calibration_error = (
        sum(confidence_errors) / total if total else 1.0
    )
    return MetricReport(
        total=total,
        action_accuracy=_ratio(action_correct, total),
        type_accuracy=_ratio(type_correct, total),
        joint_accuracy=_ratio(joint_correct, total),
        action_confusion={
            (gold_label, predicted_label): action_confusion[(gold_label, predicted_label)]
            for gold_label in ACTIONS
            for predicted_label in ACTIONS
        },
        type_confusion={
            (gold_label, predicted_label): type_confusion[(gold_label, predicted_label)]
            for gold_label in MESSAGE_TYPES
            for predicted_label in MESSAGE_TYPES
        },
        catastrophic_action_count=catastrophic_action_count,
        catastrophic_scam_count=catastrophic_scam_count,
        miss_count=miss_count,
        evidence_precision=evidence_precision,
        evidence_recall=evidence_recall,
        evidence_f1=evidence_f1,
        calibration_error=calibration_error,
        reliability=_reliability_bins(reliability_samples),
        gold_actions={label: gold_actions[label] for label in ACTIONS},
        predicted_actions={label: predicted_actions[label] for label in ACTIONS},
        gold_types={label: gold_types[label] for label in MESSAGE_TYPES},
        predicted_types={label: predicted_types[label] for label in MESSAGE_TYPES},
        missing_prediction_ids=missing,
        extra_prediction_ids=extra,
    )


def _matrix_lines(
    title: str,
    labels: Sequence[str],
    matrix: Mapping[tuple[str, str], int],
) -> list[str]:
    width = max(7, max(len(label) for label in labels) + 1)
    lines = [title + " (rows=gold, columns=predicted)"]
    lines.append("gold\\pred".ljust(width) + "".join(label.rjust(width) for label in labels))
    for gold_label in labels:
        lines.append(
            gold_label.ljust(width)
            + "".join(
                str(matrix[(gold_label, predicted_label)]).rjust(width)
                for predicted_label in labels
            )
        )
    return lines


def _distribution_lines(
    title: str,
    labels: Sequence[str],
    gold: Mapping[str, int],
    predicted: Mapping[str, int],
    total: int,
) -> list[str]:
    lines = [title, f"{'label':<20} {'gold':>9} {'predicted':>11} {'delta_pp':>10}"]
    for label in labels:
        gold_share = 100.0 * gold[label] / total if total else 0.0
        predicted_share = 100.0 * predicted[label] / total if total else 0.0
        lines.append(
            f"{label:<20} {gold_share:>8.2f}% {predicted_share:>10.2f}% "
            f"{predicted_share - gold_share:>+9.2f}"
        )
    return lines


def print_metric_report(report: MetricReport) -> None:
    lines = [
        "",
        "=" * 100,
        f"LABELLED METRICS ({report.total} rows)",
        f"action accuracy       {report.action_accuracy:.4f}",
        f"message_type accuracy {report.type_accuracy:.4f}",
        f"joint accuracy        {report.joint_accuracy:.4f}",
        f"evidence precision    {report.evidence_precision:.4f}",
        f"evidence recall       {report.evidence_recall:.4f}",
        f"evidence F1           {report.evidence_f1:.4f}",
        f"confidence MAE        {report.calibration_error:.4f} (predicted vs gold)",
    ]
    if report.missing_prediction_ids:
        lines.append("missing predictions     " + ", ".join(report.missing_prediction_ids))
    if report.extra_prediction_ids:
        lines.append("extra predictions       " + ", ".join(report.extra_prediction_ids))
    lines.extend(
        [
            "",
            "DANGEROUS CELLS",
            "CATASTROPHIC CELL gold action=mute -> predicted action=notify: "
            f"{report.catastrophic_action_count}",
            "CATASTROPHIC SCAM SUBSET gold action=mute, gold type=scam -> "
            f"predicted action=notify: {report.catastrophic_scam_count}",
            "MISS CELL gold action=notify -> predicted action=mute: "
            f"{report.miss_count}",
            "",
        ]
    )
    lines.extend(_matrix_lines("ACTION CONFUSION 3x3", ACTIONS, report.action_confusion))
    lines.append("")
    lines.extend(
        _matrix_lines(
            "MESSAGE_TYPE CONFUSION 11x11", MESSAGE_TYPES, report.type_confusion
        )
    )
    lines.extend(
        [
            "",
            "CONFIDENCE RELIABILITY (0.05 bins)",
            f"{'bin':<13} {'n':>4} {'pred_mean':>10} {'gold_mean':>10} "
            f"{'mae':>8} {'action_acc':>11}",
        ]
    )
    for bucket in report.reliability:
        closing = "]" if bucket.upper == 1.0 else ")"
        label = f"[{bucket.lower:.2f},{bucket.upper:.2f}{closing}"
        if bucket.count == 0:
            lines.append(f"{label:<13} {0:>4} {'-':>10} {'-':>10} {'-':>8} {'-':>11}")
        else:
            lines.append(
                f"{label:<13} {bucket.count:>4} {bucket.mean_predicted:>10.4f} "
                f"{bucket.mean_gold:>10.4f} {bucket.mean_absolute_error:>8.4f} "
                f"{bucket.action_accuracy:>11.4f}"
            )
    lines.append("")
    lines.extend(
        _distribution_lines(
            "ACTION PRIOR DRIFT", ACTIONS, report.gold_actions, report.predicted_actions, report.total
        )
    )
    lines.append("")
    lines.extend(
        _distribution_lines(
            "MESSAGE_TYPE PRIOR DRIFT",
            MESSAGE_TYPES,
            report.gold_types,
            report.predicted_types,
            report.total,
        )
    )
    lines.append("=" * 100)
    print("\n".join(lines))
