"""Unit tests for the reason-quality judge. No network: the provider is always a stub."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.client import ProviderClientError  # noqa: E402
from evaluation.judge import (  # noqa: E402
    CRITERIA,
    MAX_CRITERION_SCORE,
    JudgeReport,
    ReasonJudgement,
    _fingerprint,
    _parse,
    judge_reason,
    score_rows,
)


class _StubCompletion:
    def __init__(self, text: str) -> None:
        self.response = mock.Mock(output_text=text)


class _StubProvider:
    """Counts calls so a test can prove how many model calls a batch actually made."""

    def __init__(self, text: str = '{"specificity":2,"consistency":3,"register":3,"note":"ok"}') -> None:
        self.text = text
        self.calls = 0

    def complete(self, messages, tools, model, **kw):
        self.calls += 1
        if tools:
            raise AssertionError("the judge must be called with no tools")
        return _StubCompletion(self.text)


class _FailingProvider:
    def complete(self, messages, tools, model, **kw):
        raise ProviderClientError("provider exploded", attempts=4, category="transient")


class ParseTest(unittest.TestCase):
    def test_it_accepts_a_well_formed_payload(self) -> None:
        self.assertEqual(
            _parse('{"specificity":0,"consistency":1,"register":3,"note":"n"}'),
            (0, 1, 3, "n"),
        )

    def test_it_rejects_a_score_outside_the_scale(self) -> None:
        for payload in (
            '{"specificity":4,"consistency":1,"register":1,"note":""}',
            '{"specificity":-1,"consistency":1,"register":1,"note":""}',
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    _parse(payload)

    def test_it_rejects_a_non_integer_score(self) -> None:
        # `True` is an int in Python; a boolean slipping through would score as 1.
        for payload in (
            '{"specificity":"3","consistency":1,"register":1,"note":""}',
            '{"specificity":true,"consistency":1,"register":1,"note":""}',
            '{"specificity":2.5,"consistency":1,"register":1,"note":""}',
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    _parse(payload)

    def test_it_rejects_a_missing_criterion(self) -> None:
        with self.assertRaises(ValueError):
            _parse('{"specificity":1,"consistency":1,"note":""}')


class NormalizationTest(unittest.TestCase):
    def test_a_perfect_row_normalizes_to_one(self) -> None:
        top = MAX_CRITERION_SCORE
        judgement = ReasonJudgement("m", top, top, top, "")
        self.assertEqual(judgement.total, top * len(CRITERIA))
        self.assertEqual(judgement.normalized, 1.0)

    def test_a_zero_row_normalizes_to_zero(self) -> None:
        self.assertEqual(ReasonJudgement("m", 0, 0, 0, "").normalized, 0.0)

    def test_failed_rows_are_excluded_from_the_mean(self) -> None:
        report = JudgeReport(
            (
                ReasonJudgement("a", 3, 3, 3, ""),
                ReasonJudgement("b", 0, 0, 0, "boom", failed=True),
            ),
            "test",
        )
        self.assertEqual(report.failure_count, 1)
        self.assertEqual(len(report.scored), 1)
        # A failed row scored as a zero would silently halve the reported quality.
        self.assertEqual(report.mean_normalized, 1.0)


class FailurePathTest(unittest.TestCase):
    def test_a_provider_failure_is_recorded_not_raised(self) -> None:
        judgement = judge_reason(_FailingProvider(), "m", "reason.", "mute", "spam")
        self.assertTrue(judgement.failed)
        self.assertIn("provider exploded", judgement.note)


class FingerprintTest(unittest.TestCase):
    def test_identical_inputs_share_a_fingerprint(self) -> None:
        self.assertEqual(
            _fingerprint("same text.", "mute", "spam"),
            _fingerprint("same text.", "mute", "spam"),
        )

    def test_the_action_and_type_are_part_of_the_key(self) -> None:
        base = _fingerprint("same text.", "mute", "spam")
        self.assertNotEqual(base, _fingerprint("same text.", "digest", "spam"))
        self.assertNotEqual(base, _fingerprint("same text.", "mute", "promotion"))


class BatchTest(unittest.TestCase):
    def test_identical_rows_are_scored_once_and_fanned_out(self) -> None:
        """Gold reasons repeat verbatim; scoring each copy separately raced and double-spent."""
        rows = [
            ("a", "The message is a harmless greeting.", "digest", "greeting"),
            ("b", "The message is a harmless greeting.", "digest", "greeting"),
            ("c", "A different sentence entirely here.", "mute", "spam"),
        ]
        provider = _StubProvider()
        with mock.patch("evaluation.judge._load_cache", return_value={}), mock.patch(
            "evaluation.judge._append_cache"
        ):
            report = score_rows(rows, label="t", workers=3, provider=provider)

        self.assertEqual(provider.calls, 2, "two distinct reasons should cost two calls")
        self.assertEqual(len(report.judgements), 3, "every input row still gets a judgement")
        self.assertEqual([item.message_id for item in report.judgements], ["a", "b", "c"])
        by_id = {item.message_id: item.total for item in report.judgements}
        self.assertEqual(by_id["a"], by_id["b"], "identical rows must score identically")

    def test_a_cache_hit_makes_no_model_call(self) -> None:
        rows = [("a", "Some reason sentence.", "notify", "urgent")]
        fingerprint = _fingerprint("Some reason sentence.", "notify", "urgent")
        cache = {
            fingerprint: {"specificity": 1, "consistency": 2, "register": 3, "note": "cached"}
        }
        provider = _StubProvider()
        with mock.patch("evaluation.judge._load_cache", return_value=cache):
            report = score_rows(rows, label="t", workers=1, provider=provider)

        self.assertEqual(provider.calls, 0)
        self.assertEqual(report.judgements[0].total, 6)
        self.assertEqual(report.judgements[0].note, "cached")


if __name__ == "__main__":
    unittest.main()
