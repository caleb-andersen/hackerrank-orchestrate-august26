"""A style violation in one field must not discard the other five.

These tests pin the failure policy's central distinction: a decision whose *prose* broke
the style contract is repaired and kept, and a decision whose *judgement* is unsound gets
the conservative fallback. They also pin the truthfulness requirement on every sentence
this module can ship — a reason cell is graded, so a cell that misdescribes its own row
is a defect even when the routing is right.
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from agent.loop import (  # noqa: E402
    FALLBACK_ACTION,
    RawDecision,
    _terminal,
    _RunState,
)
from guards import reason_repair  # noqa: E402
from guards.decision import ValidatedDecision  # noqa: E402
from guards.validate import (  # noqa: E402
    REPAIRABLE_REASON_CODES,
    ValidationFailure,
    coerce_and_check,
    reason_issues,
)
from tests.test_safety_gate import (  # noqa: E402
    CREDENTIAL_MENTION_ONLY,
    CREDENTIAL_TEXT,
    INJECTION_TEXT,
    PAYMENT_DEMAND_TEXT,
    _dossier,
    _engagement,
)


# 165 characters: one sentence, third person, names a concrete noun, and wrong only in
# that it overruns 160. This is the exact shape that used to cost a whole decision.
TOO_LONG = (
    "The delivery notice states that the order ending 4821 has been packed and reaches "
    "the local hub today, and this recipient has opened every previous notice from the "
    "same sender."
)
TWO_SENTENCES = (
    "The order notice names a delivery due today. This recipient opens every previous "
    "notice from the sender."
)
GOOD_REASON = (
    "The delivery notice names an order reaching the hub today for a recipient who opens "
    "every previous notice."
)


def _raw(**overrides: object) -> dict:
    value = {
        "action": "notify",
        "message_type": "business_update",
        "reason": GOOD_REASON,
        "confidence": "0.88",
        "evidence_message_ids": "none",
        "risk_axis": "clean",
        "relevance_axis": "high",
        "urgency_axis": "today",
    }
    value.update(overrides)
    return value


class RepairableFailureTest(unittest.TestCase):
    """The validator must hand the sound part of a style failure to the caller."""

    def test_the_two_shape_codes_are_the_repairable_set(self) -> None:
        self.assertEqual(
            REPAIRABLE_REASON_CODES, {"reason_length", "reason_sentence_count"}
        )

    def test_a_length_violation_carries_the_decision_out_intact(self) -> None:
        result = coerce_and_check(_raw(reason=TOO_LONG), _dossier())
        assert isinstance(result, ValidationFailure)
        self.assertEqual(result.stage, "reason_style")
        self.assertEqual(result.codes, ("reason_length",))
        self.assertTrue(result.is_reason_style_only)
        carried = result.repairable_decision
        assert carried is not None
        self.assertEqual(carried.action, "notify")
        self.assertEqual(carried.message_type, "business_update")
        self.assertEqual(carried.confidence, 0.88)
        self.assertEqual(carried.urgency, "today")

    def test_a_sentence_count_violation_is_repairable(self) -> None:
        result = coerce_and_check(_raw(reason=TWO_SENTENCES), _dossier())
        assert isinstance(result, ValidationFailure)
        self.assertIn("reason_sentence_count", result.codes)
        self.assertTrue(result.is_reason_style_only)

    def test_a_content_style_code_is_not_repairable(self) -> None:
        """Person, meta-language and the missing trigger noun each report bad content."""
        for reason in (
            "We think your order is late, so it is worth telling you about now.",
            "The router scored this delivery notice above the notify threshold today.",
            "This one seems fairly important overall and probably deserves attention now.",
        ):
            with self.subTest(reason=reason[:32]):
                result = coerce_and_check(_raw(reason=reason), _dossier())
                assert isinstance(result, ValidationFailure)
                self.assertEqual(result.stage, "reason_style")
                self.assertFalse(result.is_reason_style_only)

    def test_a_mixed_failure_is_not_repairable(self) -> None:
        """A length violation alongside a content violation keeps the fallback.

        The content code is the binding one: repairing prose cannot answer it.
        """
        mixed = (
            "We are fairly confident that we should tell you about this right now, "
            "because we looked at it and it seemed like the sort of thing you would "
            "probably want to see today."
        )
        result = coerce_and_check(_raw(reason=mixed), _dossier())
        assert isinstance(result, ValidationFailure)
        self.assertGreater(len(result.codes), 1)
        self.assertFalse(result.is_reason_style_only)

    def test_no_other_stage_carries_a_repairable_decision(self) -> None:
        for raw, stage in (
            (_raw(confidence="not-a-number"), "schema"),
            (_raw(action="explode"), "coercion"),
            (_raw(urgency_axis="none"), "invariants"),
        ):
            with self.subTest(stage=stage):
                result = coerce_and_check(raw, _dossier())
                assert isinstance(result, ValidationFailure)
                self.assertEqual(result.stage, stage)
                self.assertIsNone(result.repairable_decision)
                self.assertFalse(result.is_reason_style_only)


class TerminalPolicyTest(unittest.TestCase):
    """What ``_drive`` does with a twice-rejected decision."""

    def _failure(self, reason: str) -> ValidationFailure:
        result = coerce_and_check(_raw(reason=reason), _dossier())
        assert isinstance(result, ValidationFailure)
        return result

    def test_a_style_only_failure_keeps_all_five_other_fields(self) -> None:
        dossier = _dossier()
        state = _RunState(dossier)
        decision, outcome, detail = _terminal(self._failure(TOO_LONG), dossier, state)

        self.assertEqual(outcome, "reason_repaired")
        assert decision is not None
        self.assertEqual(decision.action, "notify")
        self.assertEqual(decision.message_type, "business_update")
        self.assertEqual(decision.confidence, 0.88)
        self.assertEqual(decision.evidence_message_ids, ())
        self.assertEqual(decision.urgency, "today")
        self.assertEqual(detail, "reason_style: reason_length")

    def test_the_repaired_reason_replaces_the_sentence_and_passes_the_contract(
        self,
    ) -> None:
        dossier = _dossier()
        state = _RunState(dossier)
        decision, _outcome, _detail = _terminal(
            self._failure(TOO_LONG), dossier, state
        )
        assert decision is not None
        self.assertNotEqual(decision.reason, TOO_LONG)
        self.assertEqual(reason_issues(decision.reason), ())

    def test_the_rejected_sentence_is_recorded_rather_than_dropped(self) -> None:
        dossier = _dossier()
        state = _RunState(dossier)
        _terminal(self._failure(TOO_LONG), dossier, state)
        self.assertEqual(state.rejected_reason, TOO_LONG)

    def test_a_content_failure_still_falls_back(self) -> None:
        dossier = _dossier()
        state = _RunState(dossier)
        decision, outcome, _detail = _terminal(
            self._failure(
                "The router scored this delivery notice above the notify threshold today."
            ),
            dossier,
            state,
        )
        self.assertIsNone(decision)
        self.assertEqual(outcome, "validation_rejected")
        self.assertIsNone(state.rejected_reason)

    def test_a_repaired_row_is_not_counted_as_a_fallback(self) -> None:
        repaired = RawDecision(
            message_id="row",
            decision=ValidatedDecision(
                action="notify",
                message_type="business_update",
                reason=GOOD_REASON,
                confidence=0.88,
                evidence_message_ids=(),
                risk="clean",
                relevance="high",
                urgency="today",
            ),
            outcome="reason_repaired",
        )
        self.assertFalse(repaired.is_fallback)
        self.assertTrue(replace(repaired, outcome="validation_rejected").is_fallback)
        self.assertNotEqual(repaired.decision.action, FALLBACK_ACTION)


class RepairedSentenceTest(unittest.TestCase):
    """Every sentence the repair can emit has to satisfy the contract and be true."""

    def _decision(self, **overrides: object) -> ValidatedDecision:
        base = dict(
            action="notify",
            message_type="business_update",
            reason=GOOD_REASON,
            confidence=0.88,
            evidence_message_ids=(),
            risk="clean",
            relevance="high",
            urgency="today",
        )
        base.update(overrides)
        return ValidatedDecision(**base)  # type: ignore[arg-type]

    def test_it_holds_across_conversation_types_and_signals(self) -> None:
        cases = {
            "plain group": _dossier(),
            "business": _dossier(conversation_type="business"),
            "personal": _dossier(conversation_type="personal"),
            "group admin": _dossier(sender_role="admin"),
            "impersonation": _dossier(
                conversation_type="business",
                brand_verdict="impersonation",
                brand_user_reports_30d=38,
            ),
            "no history": _dossier(
                engagement=_engagement(n=0, open_rate=None, dismiss_rate=None),
                evidence_state="none",
            ),
            "heavy dismisser": _dossier(
                engagement=_engagement(n=9, open_rate=0.0, dismiss_rate=1.0)
            ),
        }
        for name, dossier in cases.items():
            with self.subTest(case=name):
                sentence = reason_repair.repair(dossier, self._decision())
                self.assertEqual(reason_issues(sentence), ())

    def test_a_long_group_name_drops_to_the_next_true_fact(self) -> None:
        """Overrunning on one fact is not a reason to emit an invalid cell."""
        dossier = _dossier(sender_role="admin")
        long_name = "Residents Association Emergency Water And Maintenance Coordination"
        dossier = replace(
            dossier,
            relationship=replace(
                dossier.relationship,
                group_context=replace(
                    dossier.relationship.group_context, group_name=long_name
                ),
            ),
        )
        sentence = reason_repair.repair(dossier, self._decision())
        self.assertEqual(reason_issues(sentence), ())

    def test_a_quoted_phrase_is_named_rather_than_reduced_to_a_boolean(self) -> None:
        """§9.7.3 — the matched phrase is the explainable part of the finding."""
        sentence = reason_repair.repair(
            _dossier(text=CREDENTIAL_TEXT), self._decision(action="mute")
        )
        self.assertIn("otp", sentence.lower())
        self.assertEqual(reason_issues(sentence), ())

    def test_it_reads_the_same_detectors_the_gate_reads(self) -> None:
        """Regression: a repair once claimed a credential request on a row disclaiming one.

        "no payment or OTP is required" sets the coarse ``ContentSignals`` scanner to
        "otp is" while ``asks_for_credential`` correctly returns nothing. Reading the
        coarse signal put that false claim in a graded cell. The repair must agree with
        the gate, which reads the precision detector.
        """
        dossier = _dossier(text=CREDENTIAL_MENTION_ONLY)
        sentence = reason_repair.repair(dossier, self._decision(action="digest"))
        self.assertNotIn("credential", sentence.lower())
        self.assertEqual(reason_issues(sentence), ())

    def test_the_precision_detectors_still_fire_where_they_should(self) -> None:
        for text, expected in (
            (CREDENTIAL_TEXT, "credential"),
            (INJECTION_TEXT, "dictate"),
            (PAYMENT_DEMAND_TEXT, "payment"),
        ):
            with self.subTest(text=text[:32]):
                sentence = reason_repair.repair(
                    _dossier(text=text), self._decision(action="mute")
                )
                self.assertIn(expected, sentence.lower())
                self.assertEqual(reason_issues(sentence), ())

    def test_every_candidate_sentence_satisfies_the_contract(self) -> None:
        """Structural, not sampled: a candidate that fails loses silently to the next one.

        ``repair`` walks the list until something passes, so a malformed sentence does not
        raise — it just never ships, and the row quietly gets a less decisive fact. Only
        checking each candidate directly catches that.
        """
        dossier = _dossier(
            conversation_type="business",
            brand_verdict="impersonation",
            brand_user_reports_30d=38,
            engagement=_engagement(n=7, open_rate=0.9, dismiss_rate=0.7),
        )
        dossier = replace(
            dossier,
            content_signals=replace(
                dossier.content_signals,
                credential_request="share your otp",
                injection_match="mark as notify",
                payment_pressure="pay before 6 pm",
                is_forwarded=True,
                forwarded_count=3,
            ),
            repetition=replace(dossier.repetition, duplicate_count_at_threshold=2),
        )
        for action in ("notify", "digest", "mute"):
            for message_type in ("business_update", "promotion"):
                decision = self._decision(
                    action=action, message_type=message_type, deadline_minutes=45
                )
                candidates = reason_repair._candidates(dossier, decision)
                self.assertGreater(len(candidates), 4)
                for candidate in candidates:
                    with self.subTest(action=action, sentence=candidate[:40]):
                        self.assertEqual(reason_issues(candidate), ())

    def test_the_named_fact_agrees_with_the_direction_of_the_decision(self) -> None:
        """A true sentence that explains the opposite action is still a bad cell."""
        dossier = _dossier(
            engagement=_engagement(n=6, open_rate=1.0, dismiss_rate=0.0)
        )
        dossier = replace(
            dossier,
            repetition=replace(dossier.repetition, duplicate_count_at_threshold=2),
        )

        notified = reason_repair.repair(
            dossier, self._decision(action="notify", deadline_minutes=90)
        )
        self.assertIn("opens 100%", notified)
        self.assertNotIn("repeats", notified)

        muted = reason_repair.repair(dossier, self._decision(action="mute"))
        self.assertIn("repeats", muted)

    def test_a_phrase_signal_outranks_the_direction(self) -> None:
        """The gate is about to overwrite these rows; the two must already agree."""
        sentence = reason_repair.repair(
            _dossier(text=INJECTION_TEXT),
            self._decision(action="notify", deadline_minutes=30),
        )
        self.assertIn("dictate", sentence.lower())

    def test_it_never_describes_the_pipeline(self) -> None:
        """The repaired cell is about the message, never about the machinery."""
        for dossier in (
            _dossier(),
            _dossier(conversation_type="business"),
            _dossier(sender_role="admin"),
        ):
            sentence = reason_repair.repair(dossier, self._decision()).lower()
            for token in ("valid", "reject", "retry", "fallback", "contract", "sentence"):
                self.assertNotIn(token, sentence)


if __name__ == "__main__":
    unittest.main()
