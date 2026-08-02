import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

"""Run labelled samples, report every scored field, and gate the full submission."""

import argparse  # noqa: E402
import logging  # noqa: E402
from typing import Sequence  # noqa: E402

from config import (  # noqa: E402
    DATASET_DIR,
    DECISION_MODEL_DEV,
    MAX_CONCURRENCY,
    OUTPUT_PATH,
)
from agent.client import ProviderClientError  # noqa: E402
from evaluation.consistency import (  # noqa: E402
    audit_full_predictions,
    print_consistency_report,
)
from evaluation.judge import (  # noqa: E402
    gold_rows,
    prediction_rows,
    print_judge_report,
    score_rows,
)
from evaluation.metrics import evaluate_metrics, print_metric_report  # noqa: E402
from evaluation.records import load_gold_samples, load_predictions  # noqa: E402
from evaluation.validate_output import (  # noqa: E402
    print_validation_report,
    validate_output,
)


# The reason-quality term used when --no-judge is passed or the judge cannot be reached.
#
# It is 0.0 rather than the old 1.0 stub because the failure this replaces was a metric being
# claimed rather than measured: a stub of 1.0 awarded a full 15% of the composite for a column
# nothing had looked at, and it flattered every score printed before this module existed. Zero
# cannot be mistaken for a measurement. The composite line always states which term it used, so
# an unjudged run is visibly 0.15 lower than a judged one rather than quietly comparable to it.
UNMEASURED_REASON_QUALITY = 0.0


def composite_score(
    action_accuracy: float,
    type_accuracy: float,
    evidence_f1: float,
    reason_quality: float,
    calibration_error: float,
) -> float:
    return (
        0.35 * action_accuracy
        + 0.25 * type_accuracy
        + 0.15 * evidence_f1
        + 0.15 * reason_quality
        + 0.10 * (1.0 - calibration_error)
    )


def _run_samples(args: argparse.Namespace, predictions_path: pathlib.Path) -> int:
    """Pass only unlabelled Message objects across the production-runner boundary."""
    import main as router_main
    from data.loader import load_dataset

    samples_path = args.dataset / "sample_messages.csv"
    samples = load_gold_samples(samples_path)
    dataset = load_dataset(args.dataset)
    selection = router_main.Selection(
        messages=[sample.message for sample in samples],
        order=[sample.message.message_id for sample in samples],
        destination=predictions_path,
        is_full_run=False,
    )
    runner_args = argparse.Namespace(
        dry_run=False,
        no_dnd=args.no_dnd,
        resume=args.resume,
        verbose=args.verbose,
        workers=args.workers,
        model=args.model,
    )
    return router_main.run_selection(dataset, selection, runner_args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the router on labelled samples, score all fields, and audit the full output."
        )
    )
    parser.add_argument("--dataset", type=pathlib.Path, default=DATASET_DIR)
    parser.add_argument(
        "--predictions",
        type=pathlib.Path,
        help="score an existing sample prediction CSV instead of running the agent",
    )
    parser.add_argument(
        "--full-predictions",
        type=pathlib.Path,
        default=OUTPUT_PATH,
        help="full 110-row output used by the submission gate and consistency audit",
    )
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENCY)
    parser.add_argument(
        "--model",
        metavar="ID",
        default=DECISION_MODEL_DEV,
        help=(
            "decision model for the sample run. Defaults to the faster development model "
            f"({DECISION_MODEL_DEV}) because this path is re-run many times per build; "
            "the graded 110-row run defaults to the production model instead. Pass the "
            "production model here to measure what the submission will actually score."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-dnd", action="store_true")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "skip the reason-quality judge. The composite then carries "
            f"reason_quality={UNMEASURED_REASON_QUALITY:.1f} and says so, rather than "
            "substituting a value nothing measured."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    sample_path = args.dataset / "sample_messages.csv"
    predictions_path = args.predictions or args.dataset / "output.samples.csv"

    run_status = 0
    if args.predictions is None:
        run_status = _run_samples(args, predictions_path)
        if run_status:
            print("SAMPLE AGENT RUN FAILED; diagnostics below use rows that were written.")

    try:
        gold = load_gold_samples(sample_path)
        predictions = load_predictions(predictions_path)
        metric_report = evaluate_metrics(gold, predictions)
    except (OSError, ValueError) as error:
        print(f"LABELLED METRICS FAIL: {error}")
        return 1
    print_metric_report(metric_report)

    validation_report = validate_output(
        args.full_predictions,
        args.dataset / "messages.csv",
        args.dataset / "message_history.csv",
    )
    print_validation_report(validation_report)

    try:
        consistency_report = audit_full_predictions(
            args.dataset, args.full_predictions
        )
    except (OSError, ValueError) as error:
        print(f"CONSISTENCY AUDIT FAIL: {error}")
        consistency_report = None
    if consistency_report is not None:
        print_consistency_report(consistency_report)

    reason_quality = UNMEASURED_REASON_QUALITY
    reason_quality_source = "NOT MEASURED (--no-judge)"
    if not args.no_judge:
        try:
            # Gold is scored every run as the reference line. After the first run its rows are
            # content-hash cache hits, so this costs no model calls until the rubric changes —
            # which is exactly when the reference does need recomputing.
            gold_report = score_rows(
                gold_rows(args.dataset),
                label="GOLD reasons (reference line)",
                workers=args.workers,
            )
            prediction_report = score_rows(
                prediction_rows(predictions_path),
                label=f"predictions ({predictions_path})",
                workers=args.workers,
            )
        except (ProviderClientError, OSError, ValueError) as error:
            print(f"REASON JUDGE UNAVAILABLE: {type(error).__name__}: {error}")
            reason_quality_source = f"NOT MEASURED ({type(error).__name__})"
        else:
            print_judge_report(gold_report)
            print_judge_report(prediction_report)
            reason_quality = prediction_report.mean_normalized
            gold_quality = gold_report.mean_normalized
            delta = reason_quality - gold_quality
            reason_quality_source = (
                f"measured, gold reference {gold_quality:.4f}, delta {delta:+.4f}"
            )
            if prediction_report.failure_count:
                reason_quality_source += (
                    f", {prediction_report.failure_count} row(s) unscored"
                )

    score = composite_score(
        metric_report.action_accuracy,
        metric_report.type_accuracy,
        metric_report.evidence_f1,
        reason_quality,
        metric_report.calibration_error,
    )
    print(
        "COMPOSITE SCORE = "
        "0.35*action_accuracy + 0.25*type_accuracy + 0.15*evidence_f1 + "
        "0.15*reason_quality + 0.10*(1-calibration_error)"
    )
    print(f"COMPOSITE SCORE: {score:.6f} (reason_quality={reason_quality:.4f} — {reason_quality_source})")

    metric_shape_failed = bool(
        metric_report.missing_prediction_ids or metric_report.extra_prediction_ids
    )
    consistency_failed = (
        consistency_report is None
        or consistency_report.bug_count > 0
        or bool(consistency_report.missing_prediction_ids)
    )
    return int(
        bool(run_status)
        or metric_shape_failed
        or not validation_report.ok
        or consistency_failed
    )


if __name__ == "__main__":
    raise SystemExit(main())
