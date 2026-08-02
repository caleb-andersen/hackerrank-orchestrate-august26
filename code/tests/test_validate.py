"""Both-directions tests for the model-decision validation boundary."""

import logging
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import MAX_EVIDENCE_IDS  # noqa: E402
from context.features import Dossier  # noqa: E402
from context.retrieval import EvidenceCandidate  # noqa: E402
from guards.decision import ValidatedDecision  # noqa: E402
from guards.validate import (  # noqa: E402
    ValidationFailure,
    coerce_and_check,
)


VALID_REASON = (
    "The payment deadline and sender history make this reminder time-sensitive."
)


def _candidate(
    evidence_id: str,
    *,
    dismissed: bool = False,
    muted_after: bool = False,
    reported: bool = False,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        history_message_id=evidence_id,
        score=0.8,
        same_peer=True,
        same_group=False,
        jaccard=0.5,
        event_relevance=0.6,
        recency=0.9,
        created_at=datetime(2026, 7, 1, 9, 0),
        days_ago=1.0,
        conversation_type="personal",
        text_excerpt="Prior reminder",
        opened=True,
        replied=False,
        dismissed=dismissed,
        muted_after=muted_after,
        reported=reported,
    )


def _dossier(*candidates: EvidenceCandidate) -> Dossier:
    # Validation reads exactly this public Dossier field. A narrow structural stand-in
    # keeps each invariant test focused without copying the full feature-layer builder.
    return cast(Dossier, SimpleNamespace(evidence_candidates=tuple(candidates)))


def _raw(**overrides: object) -> dict:
    value = {
        "action": "digest",
        "message_type": "payment",
        "reason": VALID_REASON,
        "confidence": "0.82",
        "evidence_message_ids": "none",
        "risk_axis": "clean",
        "relevance_axis": "high",
        "urgency_axis": "today",
    }
    value.update(overrides)
    return value


def _assert_failure(result: object, code: str) -> ValidationFailure:
    if not isinstance(result, ValidationFailure):
        raise AssertionError(f"Expected ValidationFailure, got {result!r}")
    if code not in result.codes:
        raise AssertionError(f"Expected {code!r} in {result.codes!r}")
    return result


class SchemaTest(unittest.TestCase):
    def test_every_required_field_must_be_present(self) -> None:
        raw = _raw()
        del raw["confidence"]
        failure = _assert_failure(coerce_and_check(raw, _dossier()), "missing_field")
        self.assertEqual(failure.stage, "schema")
        self.assertEqual(failure.issues[0].field, "confidence")

    def test_confidence_parses_as_a_bounded_float(self) -> None:
        result = coerce_and_check(_raw(confidence="0.75"), _dossier())
        self.assertIsInstance(result, ValidatedDecision)
        assert isinstance(result, ValidatedDecision)
        self.assertEqual(result.confidence, 0.75)
        for bad in ("not-a-number", -0.01, 1.01, float("nan"), True):
            with self.subTest(confidence=bad):
                _assert_failure(
                    coerce_and_check(_raw(confidence=bad), _dossier()),
                    "invalid_confidence",
                )

    def test_evidence_uses_the_contract_string_format(self) -> None:
        for bad in ([], "", "one;", "one; two", "one two", "none;one"):
            with self.subTest(evidence=bad):
                _assert_failure(
                    coerce_and_check(
                        _raw(evidence_message_ids=bad),
                        _dossier(),
                    ),
                    "invalid_evidence_format",
                )


class CoercionTierTest(unittest.TestCase):
    def test_exact_values_are_accepted_without_coercion(self) -> None:
        result = coerce_and_check(
            _raw(action="notify", message_type="business_update"),
            _dossier(),
        )
        self.assertIsInstance(result, ValidatedDecision)
        assert isinstance(result, ValidatedDecision)
        self.assertEqual((result.action, result.message_type), ("notify", "business_update"))
        self.assertEqual(result.coercion_count, 0)

    def test_normalised_values_are_coerced_and_logged(self) -> None:
        with self.assertLogs("guards.validate", level=logging.INFO) as captured:
            result = coerce_and_check(
                _raw(action=" Notify ", message_type="Business Update"),
                _dossier(),
            )
        self.assertIsInstance(result, ValidatedDecision)
        assert isinstance(result, ValidatedDecision)
        self.assertEqual((result.action, result.message_type), ("notify", "business_update"))
        self.assertEqual(result.coercion_count, 2)
        log_text = "\n".join(captured.output)
        self.assertIn("' Notify '", log_text)
        self.assertIn("'notify'", log_text)
        self.assertIn("'Business Update'", log_text)
        self.assertIn("'business_update'", log_text)

    def test_token_subset_prefers_the_longest_vocabulary_match(self) -> None:
        result = coerce_and_check(
            _raw(
                action="please digest this later",
                message_type="urgent business account update notice",
            ),
            _dossier(),
        )
        self.assertIsInstance(result, ValidatedDecision)
        assert isinstance(result, ValidatedDecision)
        self.assertEqual(result.action, "digest")
        self.assertEqual(result.message_type, "business_update")
        self.assertEqual(result.coercion_count, 2)

    def test_an_unknown_value_never_gets_a_silent_default(self) -> None:
        failure = _assert_failure(
            coerce_and_check(_raw(action="send it somewhere"), _dossier()),
            "invalid_action",
        )
        self.assertEqual(failure.stage, "coercion")

    def test_a_successful_coercion_is_counted_even_when_the_other_field_fails(self) -> None:
        with self.assertLogs("guards.validate", level=logging.INFO):
            failure = _assert_failure(
                coerce_and_check(
                    _raw(action=" Digest ", message_type="unclassifiable thing"),
                    _dossier(),
                ),
                "invalid_message_type",
            )
        self.assertEqual(failure.coercion_count, 1)


class CrossFieldInvariantTest(unittest.TestCase):
    def test_i1_rejects_risky_rows_without_the_required_mute_and_type(self) -> None:
        for action, message_type in (("notify", "scam"), ("mute", "personal")):
            with self.subTest(action=action, message_type=message_type):
                _assert_failure(
                    coerce_and_check(
                        _raw(
                            action=action,
                            message_type=message_type,
                            risk_axis="scam_or_unsafe",
                        ),
                        _dossier(),
                    ),
                    "I1",
                )

    def test_i1_accepts_a_risky_muted_scam(self) -> None:
        result = coerce_and_check(
            _raw(
                action="mute",
                message_type="scam",
                risk_axis="scam_or_unsafe",
            ),
            _dossier(),
        )
        self.assertIsInstance(result, ValidatedDecision)

    def test_i2_rejects_notify_without_urgency(self) -> None:
        _assert_failure(
            coerce_and_check(_raw(action="notify", urgency_axis="none"), _dossier()),
            "I2",
        )

    def test_i2_accepts_non_notify_without_urgency(self) -> None:
        result = coerce_and_check(
            _raw(action="digest", urgency_axis="none"),
            _dossier(),
        )
        self.assertIsInstance(result, ValidatedDecision)

    def test_i3_rejects_a_clean_unwanted_mute_without_negative_event_evidence(self) -> None:
        candidate = _candidate("history_opened")
        _assert_failure(
            coerce_and_check(
                _raw(
                    action="mute",
                    relevance_axis="unwanted",
                    evidence_message_ids="history_opened",
                ),
                _dossier(candidate),
            ),
            "I3",
        )

    def test_i3_accepts_dismissal_evidence_and_the_non_clean_exception(self) -> None:
        for event_flag in ("dismissed", "muted_after", "reported"):
            with self.subTest(event=event_flag):
                evidence_id = f"history_{event_flag}"
                candidate = _candidate(evidence_id, **{event_flag: True})
                with_evidence = coerce_and_check(
                    _raw(
                        action="mute",
                        relevance_axis="unwanted",
                        evidence_message_ids=evidence_id,
                    ),
                    _dossier(candidate),
                )
                self.assertIsInstance(with_evidence, ValidatedDecision)
        content_justified = coerce_and_check(
            _raw(
                action="mute",
                message_type="spam",
                risk_axis="suspect",
                relevance_axis="unwanted",
            ),
            _dossier(),
        )
        self.assertIsInstance(content_justified, ValidatedDecision)

    def test_i4_drops_ids_outside_this_rows_candidate_set(self) -> None:
        candidate = _candidate("history_valid")
        result = coerce_and_check(
            _raw(evidence_message_ids="history_valid;history_unoffered"),
            _dossier(candidate),
        )
        self.assertIsInstance(result, ValidatedDecision)
        assert isinstance(result, ValidatedDecision)
        self.assertEqual(result.evidence_message_ids, ("history_valid",))

    def test_i4_preserves_ids_inside_this_rows_candidate_set(self) -> None:
        candidates = (_candidate("history_one"), _candidate("history_two"))
        result = coerce_and_check(
            _raw(evidence_message_ids="history_one;history_two"),
            _dossier(*candidates),
        )
        self.assertIsInstance(result, ValidatedDecision)
        assert isinstance(result, ValidatedDecision)
        self.assertEqual(result.evidence_message_ids, ("history_one", "history_two"))

    def test_i5_rejects_none_when_candidates_exist(self) -> None:
        _assert_failure(
            coerce_and_check(_raw(), _dossier(_candidate("history_available"))),
            "I5",
        )

    def test_i5_accepts_none_when_the_candidate_set_is_empty(self) -> None:
        self.assertIsInstance(coerce_and_check(_raw(), _dossier()), ValidatedDecision)

    def test_i6_rejects_more_than_the_configured_evidence_limit(self) -> None:
        ids = tuple(f"history_{index}" for index in range(MAX_EVIDENCE_IDS + 1))
        _assert_failure(
            coerce_and_check(
                _raw(evidence_message_ids=";".join(ids)),
                _dossier(*(_candidate(value) for value in ids)),
            ),
            "I6",
        )

    def test_i6_accepts_the_configured_evidence_limit(self) -> None:
        ids = tuple(f"history_{index}" for index in range(MAX_EVIDENCE_IDS))
        result = coerce_and_check(
            _raw(evidence_message_ids=";".join(ids)),
            _dossier(*(_candidate(value) for value in ids)),
        )
        self.assertIsInstance(result, ValidatedDecision)


class ReasonStyleTest(unittest.TestCase):
    def test_a_single_concrete_third_person_sentence_passes(self) -> None:
        self.assertIsInstance(coerce_and_check(_raw(), _dossier()), ValidatedDecision)

    def test_it_rejects_multiple_sentences(self) -> None:
        _assert_failure(
            coerce_and_check(
                _raw(
                    reason=(
                        "The payment deadline is today. The sender history supports the reminder."
                    )
                ),
                _dossier(),
            ),
            "reason_sentence_count",
        )

    def test_it_rejects_reasons_outside_the_character_bounds(self) -> None:
        for reason in (
            "The payment is due.",
            "The payment " + ("deadline " * 18) + "is documented.",
        ):
            with self.subTest(length=len(reason)):
                _assert_failure(
                    coerce_and_check(_raw(reason=reason), _dossier()),
                    "reason_length",
                )

    def test_it_rejects_first_or_second_person(self) -> None:
        for reason in (
            "I found that the payment deadline and sender history make this reminder urgent.",
            "The payment deadline asks you to review the sender's reminder before noon.",
        ):
            with self.subTest(reason=reason):
                _assert_failure(
                    coerce_and_check(_raw(reason=reason), _dossier()),
                    "reason_person",
                )

    def test_it_rejects_model_or_system_meta_language(self) -> None:
        reason = "The model identifies the payment deadline and sender history as time-sensitive."
        _assert_failure(
            coerce_and_check(_raw(reason=reason), _dossier()),
            "reason_meta_language",
        )

    def test_it_requires_a_concrete_trigger_noun(self) -> None:
        reason = "The circumstances clearly justify handling this carefully without interruption."
        _assert_failure(
            coerce_and_check(_raw(reason=reason), _dossier()),
            "reason_trigger_noun",
        )


if __name__ == "__main__":
    unittest.main()
