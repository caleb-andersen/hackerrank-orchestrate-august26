"""Focused regression tests for the offline evaluation workflow."""

import csv
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data.schema import Message  # noqa: E402
from evaluation.consistency import audit_messages  # noqa: E402
from evaluation.main import composite_score  # noqa: E402
from evaluation.metrics import evaluate_metrics  # noqa: E402
from evaluation.records import GoldSample, Prediction  # noqa: E402
from evaluation.validate_output import (  # noqa: E402
    EXPECTED_ROW_COUNT,
    OUTPUT_COLUMNS,
    validate_output,
)


def _message(identifier: str, text: str = "same normalized notification text") -> Message:
    return Message(
        message_id=identifier,
        user_id="recipient",
        conversation_type="personal",
        group_id=None,
        business_id=None,
        sender_user_id="sender",
        created_at=datetime(2026, 8, 1, 12, 0),
        message_text=text,
        media_type=None,
        media_id=None,
        forwarded_count=0,
    )


class MetricsTest(unittest.TestCase):
    def test_all_scored_columns_and_dangerous_cells_are_measured(self) -> None:
        gold = (
            GoldSample(
                _message("first"),
                "mute",
                "scam",
                "Gold reason.",
                0.9,
                frozenset({"history-a"}),
            ),
            GoldSample(
                _message("second"),
                "notify",
                "personal",
                "Gold reason.",
                0.7,
                frozenset(),
            ),
        )
        predictions = {
            "first": Prediction(
                "first",
                "notify",
                "scam",
                "Predicted reason.",
                0.8,
                frozenset({"history-a", "history-b"}),
            ),
            "second": Prediction(
                "second",
                "mute",
                "promotion",
                "Predicted reason.",
                0.4,
                frozenset(),
            ),
        }
        report = evaluate_metrics(gold, predictions)
        self.assertEqual(report.action_accuracy, 0.0)
        self.assertEqual(report.type_accuracy, 0.5)
        self.assertEqual(report.joint_accuracy, 0.0)
        self.assertEqual(report.catastrophic_action_count, 1)
        self.assertEqual(report.catastrophic_scam_count, 1)
        self.assertEqual(report.miss_count, 1)
        self.assertAlmostEqual(report.evidence_precision, 0.5)
        self.assertAlmostEqual(report.evidence_recall, 1.0)
        self.assertAlmostEqual(report.evidence_f1, 2 / 3)
        self.assertAlmostEqual(report.calibration_error, 0.2)
        self.assertEqual(len(report.action_confusion), 9)
        self.assertEqual(len(report.type_confusion), 121)
        self.assertEqual(len(report.reliability), 20)

    def test_composite_uses_the_requested_weights(self) -> None:
        score = composite_score(0.8, 0.6, 0.5, 1.0, 0.2)
        self.assertAlmostEqual(score, 0.735)


class ConsistencyTest(unittest.TestCase):
    def test_only_unexplained_action_divergence_is_a_bug(self) -> None:
        messages = (_message("left"), _message("middle"), _message("right"))
        actions = {"left": "notify", "middle": "mute", "right": "digest"}
        features = {
            "left": {"profile.dismiss_rate": 0.1},
            "middle": {"profile.dismiss_rate": 0.1},
            "right": {"profile.dismiss_rate": 0.9},
        }
        report = audit_messages(messages, actions, features)
        self.assertEqual(report.cluster_count, 1)
        self.assertEqual(len(report.divergent_pairs), 3)
        self.assertEqual(report.bug_count, 1)
        unexplained = [item for item in report.divergent_pairs if item.is_bug]
        self.assertEqual((unexplained[0].left_id, unexplained[0].right_id), ("left", "middle"))


class SubmissionGateTest(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _fixture(self, root: Path) -> tuple[Path, Path, Path, list[dict[str, str]]]:
        identifiers = [f"item-{index:03d}" for index in range(EXPECTED_ROW_COUNT)]
        messages = root / "messages.csv"
        history = root / "message_history.csv"
        output = root / "output.csv"
        self._write_csv(messages, ("message_id",), [{"message_id": value} for value in identifiers])
        self._write_csv(history, ("message_id",), [{"message_id": "history-a"}])
        rows = [
            {
                "message_id": identifier,
                "action": "digest",
                "message_type": "personal",
                "reason": "The sender history supports holding this ordinary update for later.",
                "confidence": "0.75",
                "evidence_message_ids": "none",
            }
            for identifier in identifiers
        ]
        self._write_csv(output, OUTPUT_COLUMNS, rows)
        return output, messages, history, rows

    def test_complete_valid_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, messages, history, _rows = self._fixture(Path(directory))
            self.assertTrue(validate_output(output, messages, history).ok)

    def test_invalid_cells_are_reported_with_the_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, messages, history, rows = self._fixture(Path(directory))
            rows[0]["action"] = "interrupt"
            rows[0]["confidence"] = "nan"
            rows[0]["reason"] = "first line\nsecond line"
            rows[0]["evidence_message_ids"] = "missing-history"
            self._write_csv(output, OUTPUT_COLUMNS, rows)
            report = validate_output(output, messages, history)
            self.assertFalse(report.ok)
            errors = " ".join(report.row_failures[0].errors)
            self.assertIn("invalid action", errors)
            self.assertIn("confidence outside", errors)
            self.assertIn("embedded newline", errors)
            self.assertIn("unresolved evidence", errors)


if __name__ == "__main__":
    unittest.main()
