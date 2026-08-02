import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

"""Split evidence precision by citation position.

``evaluate_metrics`` compares evidence as sets, so it cannot say *where* a false positive
sits. The hypothesis this answers is that the model treats ``MAX_EVIDENCE_IDS`` as a target
rather than a ceiling: if that is true, the first cited id is right far more often than the
headline precision suggests, and the false positives pile up in the second slot.
"""

import argparse  # noqa: E402
import csv  # noqa: E402
from typing import Sequence  # noqa: E402

from config import DATASET_DIR  # noqa: E402
from evaluation.records import load_gold_samples  # noqa: E402
from guards.decision import EVIDENCE_SEPARATOR, NO_EVIDENCE  # noqa: E402


def ordered_evidence(value: str) -> tuple[str, ...]:
    """Parse evidence keeping citation order, which ``parse_evidence`` discards."""
    cleaned = value.strip()
    if not cleaned or cleaned == NO_EVIDENCE:
        return ()
    return tuple(
        part.strip() for part in cleaned.split(EVIDENCE_SEPARATOR) if part.strip()
    )


def load_ordered_predictions(path: pathlib.Path) -> dict[str, tuple[str, ...]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["message_id"].strip(): ordered_evidence(row["evidence_message_ids"])
            for row in csv.DictReader(handle)
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=DATASET_DIR)
    parser.add_argument(
        "--predictions", type=pathlib.Path, default=DATASET_DIR / "output.samples.csv"
    )
    args = parser.parse_args(argv)

    gold = load_gold_samples(args.dataset / "sample_messages.csv")
    predicted = load_ordered_predictions(args.predictions)

    slot_hits: dict[int, int] = {}
    slot_total: dict[int, int] = {}
    gold_count_histogram: dict[int, int] = {}
    predicted_count_histogram: dict[int, int] = {}
    both_wrong = 0
    rows: list[str] = []

    for sample in gold:
        message_id = sample.message.message_id
        expected = sample.evidence_message_ids
        cited = predicted.get(message_id, ())
        gold_count_histogram[len(expected)] = gold_count_histogram.get(len(expected), 0) + 1
        predicted_count_histogram[len(cited)] = predicted_count_histogram.get(len(cited), 0) + 1

        marks: list[str] = []
        for slot, identifier in enumerate(cited, start=1):
            hit = identifier in expected
            slot_total[slot] = slot_total.get(slot, 0) + 1
            slot_hits[slot] = slot_hits.get(slot, 0) + int(hit)
            marks.append("hit " if hit else "MISS")
        if cited and not any(identifier in expected for identifier in cited):
            both_wrong += 1
        rows.append(
            f"{message_id:<16} gold={len(expected)} cited={len(cited)} "
            f"[{', '.join(marks) if marks else '-'}]  "
            f"cited_ids={EVIDENCE_SEPARATOR.join(cited) if cited else NO_EVIDENCE}"
        )

    lines = ["", "=" * 100, f"EVIDENCE BY CITATION POSITION ({len(gold)} labelled rows)", ""]
    lines.extend(rows)
    lines.extend(["", "PRECISION BY SLOT"])
    for slot in sorted(slot_total):
        hits = slot_hits.get(slot, 0)
        attempts = slot_total[slot]
        lines.append(
            f"  slot {slot}: {hits}/{attempts} correct = {hits / attempts:.4f}"
        )
    lines.append("")
    lines.append(
        "GOLD CITATION COUNTS       "
        + ", ".join(f"{count} id(s): {n} rows" for count, n in sorted(gold_count_histogram.items()))
    )
    lines.append(
        "PREDICTED CITATION COUNTS  "
        + ", ".join(
            f"{count} id(s): {n} rows" for count, n in sorted(predicted_count_histogram.items())
        )
    )
    lines.append(f"ROWS WHERE NO CITED ID WAS CORRECT: {both_wrong}")
    lines.append("=" * 100)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
