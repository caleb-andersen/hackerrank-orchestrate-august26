"""Validate and reconcile one model-authored routing decision.

Validation is deliberately a boundary, not an exception source.  Schema failures,
unrecognised vocabulary and cross-field contradictions are returned as structured
``ValidationFailure`` values so the caller can decide whether to retry or fall back.
The only values repaired here are ``action`` and ``message_type``; neither ever receives
a silent default.

The layers run in this order: schema, vocabulary coercion, cross-field invariants, then
reason style.  Dataset-derived evidence is compared only with the current dossier's
candidate set and is never interpreted as an instruction.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, cast

from config import MAX_EVIDENCE_IDS
from context.features import Dossier
from data.schema import (
    ACTIONS,
    MESSAGE_TYPES,
    RELEVANCE_AXES,
    RISK_AXES,
    URGENCY_AXES,
)
from guards.decision import (
    NO_EVIDENCE,
    Action,
    Relevance,
    RiskVerdict,
    Urgency,
    ValidatedDecision,
)


LOGGER = logging.getLogger(__name__)

REQUIRED_FIELDS: tuple[str, ...] = (
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
    "risk_axis",
    "relevance_axis",
    "urgency_axis",
)

# The prompt-facing axes are slightly more descriptive than the internal gate record.
# Legacy internal spellings remain accepted so this boundary composes with decisions
# produced before the prompt-facing names were introduced.
_RISK_AXIS_TO_DECISION: Mapping[str, str] = {
    "clean": "clean",
    "suspect": "suspect",
    "suspicious": "suspect",
    "unsafe": "unsafe",
    "scam_or_unsafe": "unsafe",
}
_RELEVANCE_AXIS_TO_DECISION: Mapping[str, str] = {
    "high": "high",
    "wanted": "high",
    "relevant": "high",
    "medium": "medium",
    "neutral": "medium",
    "low": "low",
    "unwanted": "low",
}
_URGENCY_AXES: frozenset[str] = frozenset(URGENCY_AXES)
_RISK_REQUIRES_MUTE: frozenset[str] = frozenset(("unsafe", "scam_or_unsafe"))
_UNWANTED_RELEVANCE: frozenset[str] = frozenset(("low", "unwanted"))


def _assert_axes_accepted() -> None:
    """Fail at import if a tool schema could emit an axis this boundary cannot map.

    ``agent.tools`` generates its enums from the same ``data.schema`` tuples read here,
    so a value added there and not mapped here would produce decisions the model was
    told to emit and the validator then rejected on every single row.
    """
    for name, vocabulary, accepted in (
        ("risk_axis", RISK_AXES, _RISK_AXIS_TO_DECISION),
        ("relevance_axis", RELEVANCE_AXES, _RELEVANCE_AXIS_TO_DECISION),
        ("urgency_axis", URGENCY_AXES, _URGENCY_AXES),
    ):
        missing = sorted(set(vocabulary) - set(accepted))
        if missing:
            raise ValueError(f"{name} vocabulary is not accepted by the validator: {missing}")


_assert_axes_accepted()

# A reason has to identify something observable, not merely say that a classification is
# appropriate.  Both singular and plural forms are explicit to keep matching readable.
CONCRETE_TRIGGER_NOUNS: frozenset[str] = frozenset(
    {
        "account",
        "admin",
        "appointment",
        "attachment",
        "bill",
        "business",
        "code",
        "credential",
        "deadline",
        "delivery",
        "discount",
        "dismissal",
        "domain",
        "event",
        "fee",
        "forward",
        "greeting",
        "group",
        "history",
        "image",
        "invoice",
        "link",
        "meeting",
        "notice",
        "offer",
        "order",
        "otp",
        "password",
        "payment",
        "pin",
        "promotion",
        "reminder",
        "reply",
        "report",
        "request",
        "sale",
        "sender",
        "transcript",
        "voice",
    }
)

_FIRST_PERSON_PRONOUNS: frozenset[str] = frozenset(
    ("i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves")
)
_SECOND_PERSON_PRONOUNS: frozenset[str] = frozenset(
    ("you", "your", "yours", "yourself", "yourselves")
)
_META_LANGUAGE: frozenset[str] = frozenset(
    (
        "ai",
        "algorithm",
        "assistant",
        "classifier",
        "model",
        "prompt",
        "router",
        "system",
        "validator",
    )
)
_WORD_PATTERN = re.compile(r"[a-z0-9]+")

# The one-sentence rule used to be `[^.!?\r\n]+[.!?]` matched with `fullmatch`, which treats
# every "." as a sentence boundary. That rejects the reasons the style contract actually asks
# for: REASON_CONTRACT tells the model to name the concrete detail that decided the row, and
# the concrete details in this dataset are amounts ("Rs. 2,500"), times ("4.30 p.m."), titles
# ("Dr. Rao"), order references ("order no. 4821") and decimals ("0.75"). Each one contains a
# period that ends nothing. A rejected reason is retried and then falls back to the
# conservative default, so the validator was converting the best-evidenced rows into
# digest/unknown — the guard was penalising compliance with the prompt.
#
# The contract is unchanged: exactly one sentence, terminal punctuation, 60–160 characters.
# Only the counter is fixed. A period is a sentence boundary unless it is one of the
# non-terminal forms below, so these are masked out of the body before the body is checked
# for any remaining terminator. Masking is deliberately conservative — an unlisted
# abbreviation is still read as a boundary and still rejected, which fails toward the old
# behaviour rather than toward silently accepting two sentences.
_DECIMAL_POINT = re.compile(r"\d\.(?=\d)")
# Written to match both "p.m." mid-sentence and the "p.m" left behind once a reason that
# ends on the abbreviation has had its terminator split off.
_CLOCK_MERIDIEM = re.compile(r"\b[ap]\.m?\.?", re.IGNORECASE)
_INITIAL = re.compile(r"\b[A-Z]\.")
_ABBREVIATION = re.compile(
    r"\b(?:Rs|USD|INR|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|No|Nos|approx|est|vs|etc"
    r"|Inc|Ltd|Pvt|Pte|Co|Apt|Ext|Ref|Fig|Ave|Rd|Blvd)\.",
    re.IGNORECASE,
)
_NON_TERMINAL_PERIODS = (_DECIMAL_POINT, _CLOCK_MERIDIEM, _ABBREVIATION, _INITIAL)
_TERMINAL_PUNCTUATION = ".!?"


def is_single_sentence(reason: str) -> bool:
    """Report whether ``reason`` is exactly one sentence ending in terminal punctuation.

    The final character is checked first and then excluded from the body scan, so a reason
    that legitimately ends on an abbreviation ("...delivered by 6 p.m.") keeps its
    terminator instead of having it masked away.
    """
    if reason != reason.strip() or "\r" in reason or "\n" in reason:
        return False
    if not reason or reason[-1] not in _TERMINAL_PUNCTUATION:
        return False
    body = reason[:-1]
    for pattern in _NON_TERMINAL_PERIODS:
        body = pattern.sub(lambda match: "_" * len(match.group()), body)
    return not any(character in _TERMINAL_PUNCTUATION for character in body)


# A double-quoted span in a reason is quoted evidence — the phrase that fired a scanner,
# the words an injection attempt used. §9.7.3 asks for exactly that phrase rather than a
# boolean, and the phrase is frequently second-person ("share your OTP") or names the
# machinery it is trying to address. Judging authorship on quoted text would therefore
# reject the reasons that carry the most evidence, so person and meta-language are
# checked against the reason with its quotations removed. Length, sentence count and the
# concrete-noun requirement still see the whole sentence.
_QUOTED_SPAN = re.compile(r"[\"“][^\"”]*[\"”]")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One machine-readable reason a raw decision cannot be accepted."""

    code: str
    message: str
    field: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    """Exception-free validation result returned to the retry/fallback caller."""

    stage: Literal["schema", "coercion", "invariants", "reason_style"]
    issues: tuple[ValidationIssue, ...]
    coercion_count: int = 0
    dropped_evidence_message_ids: tuple[str, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        """Expose compact issue codes for retry telemetry and tests."""

        return tuple(issue.code for issue in self.issues)


ValidationResult = ValidatedDecision | ValidationFailure


def _issue(code: str, message: str, field: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, field=field)


def _words(value: str) -> frozenset[str]:
    return frozenset(_WORD_PATTERN.findall(value.casefold()))


def _normalise_vocabulary_value(value: str) -> str:
    return "_".join(value.strip().casefold().split())


def _coerce_vocabulary(
    field: str,
    value: object,
    vocabulary: Sequence[str],
) -> tuple[str | None, str | None, ValidationIssue | None]:
    """Apply exact, normalised and longest token-subset matching, in that order."""

    if not isinstance(value, str):
        return None, None, _issue(
            f"invalid_{field}",
            f"{field} must be a string from the declared vocabulary.",
            field,
        )

    if value in vocabulary:
        return value, "exact", None

    normalised = _normalise_vocabulary_value(value)
    if normalised in vocabulary:
        return normalised, "normalised", None

    supplied_tokens = _words(value)
    matches: list[tuple[tuple[int, int], str]] = []
    for candidate in vocabulary:
        candidate_tokens = _words(candidate)
        if candidate_tokens and candidate_tokens.issubset(supplied_tokens):
            matches.append(((len(candidate_tokens), len(candidate)), candidate))

    if not matches:
        return None, None, _issue(
            f"invalid_{field}",
            f"{field} does not contain a recognised vocabulary value: {value!r}.",
            field,
        )

    best_score = max(score for score, _candidate in matches)
    best = sorted(candidate for score, candidate in matches if score == best_score)
    if len(best) != 1:
        return None, None, _issue(
            f"ambiguous_{field}",
            f"{field} matches equally specific values: {', '.join(best)}.",
            field,
        )
    return best[0], "token_subset", None


def _log_coercion(field: str, original: object, coerced: str, tier: str) -> None:
    LOGGER.info(
        "Coerced %s from %r to %r using %s matching",
        field,
        original,
        coerced,
        tier,
        extra={
            "field_name": field,
            "original_value": original,
            "coerced_value": coerced,
            "coercion_tier": tier,
        },
    )


def _parse_confidence(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(cast(float | int | str, value))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and 0.0 <= parsed <= 1.0 else None


def _parse_evidence(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, str):
        return None
    if value == NO_EVIDENCE:
        return ()
    parts = tuple(value.split(";"))
    if any(
        not part
        or any(character.isspace() for character in part)
        or part == NO_EVIDENCE
        for part in parts
    ):
        return None
    return parts


def _schema_issues(raw: object) -> tuple[ValidationIssue, ...]:
    if not isinstance(raw, dict):
        return (_issue("invalid_object", "The raw decision must be a dictionary."),)

    missing = tuple(field for field in REQUIRED_FIELDS if field not in raw)
    if missing:
        return tuple(
            _issue("missing_field", f"Required field {field!r} is missing.", field)
            for field in missing
        )

    issues: list[ValidationIssue] = []
    if not isinstance(raw["reason"], str):
        issues.append(_issue("invalid_reason", "reason must be a string.", "reason"))
    if _parse_confidence(raw["confidence"]) is None:
        issues.append(
            _issue(
                "invalid_confidence",
                "confidence must parse as a finite float in [0, 1].",
                "confidence",
            )
        )
    if _parse_evidence(raw["evidence_message_ids"]) is None:
        issues.append(
            _issue(
                "invalid_evidence_format",
                'evidence_message_ids must be "none" or a semicolon-joined list.',
                "evidence_message_ids",
            )
        )

    risk_axis = raw["risk_axis"]
    if not isinstance(risk_axis, str) or risk_axis not in _RISK_AXIS_TO_DECISION:
        issues.append(
            _issue("invalid_risk_axis", "risk_axis has an unknown value.", "risk_axis")
        )
    relevance_axis = raw["relevance_axis"]
    if (
        not isinstance(relevance_axis, str)
        or relevance_axis not in _RELEVANCE_AXIS_TO_DECISION
    ):
        issues.append(
            _issue(
                "invalid_relevance_axis",
                "relevance_axis has an unknown value.",
                "relevance_axis",
            )
        )
    urgency_axis = raw["urgency_axis"]
    if not isinstance(urgency_axis, str) or urgency_axis not in _URGENCY_AXES:
        issues.append(
            _issue(
                "invalid_urgency_axis",
                "urgency_axis has an unknown value.",
                "urgency_axis",
            )
        )

    media_mismatch = raw.get("media_mismatch", False)
    if not isinstance(media_mismatch, bool):
        issues.append(
            _issue(
                "invalid_media_mismatch",
                "media_mismatch must be a boolean when supplied.",
                "media_mismatch",
            )
        )
    mismatch_reason = raw.get("media_mismatch_reason")
    if mismatch_reason is not None and not isinstance(mismatch_reason, str):
        issues.append(
            _issue(
                "invalid_media_mismatch_reason",
                "media_mismatch_reason must be a string or null.",
                "media_mismatch_reason",
            )
        )
    deadline = raw.get("deadline_minutes")
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, int) or deadline < 0
    ):
        issues.append(
            _issue(
                "invalid_deadline_minutes",
                "deadline_minutes must be a non-negative integer or null.",
                "deadline_minutes",
            )
        )
    material_harm = raw.get("material_harm", False)
    if not isinstance(material_harm, bool):
        issues.append(
            _issue(
                "invalid_material_harm",
                "material_harm must be a boolean when supplied.",
                "material_harm",
            )
        )
    return tuple(issues)


def reason_issues(reason: str) -> tuple[ValidationIssue, ...]:
    """Check one reason sentence against the style contract stated in ``prompts.py``.

    Public because the evaluation harness scores the reason column with exactly the
    checks the router enforces; two implementations of one contract would let the eval
    pass rows the pipeline rejects.
    """
    issues: list[ValidationIssue] = []
    if not 60 <= len(reason) <= 160:
        issues.append(
            _issue(
                "reason_length",
                "reason must contain between 60 and 160 characters.",
                "reason",
            )
        )
    if not is_single_sentence(reason):
        issues.append(
            _issue(
                "reason_sentence_count",
                "reason must be exactly one sentence with terminal punctuation.",
                "reason",
            )
        )

    words = _words(reason)
    authored = _words(_QUOTED_SPAN.sub(" ", reason))
    first_person = sorted(authored & _FIRST_PERSON_PRONOUNS)
    second_person = sorted(authored & _SECOND_PERSON_PRONOUNS)
    if first_person or second_person:
        found = first_person + second_person
        issues.append(
            _issue(
                "reason_person",
                f"reason must be third person; found: {', '.join(found)}.",
                "reason",
            )
        )
    meta = sorted(authored & _META_LANGUAGE)
    if meta:
        issues.append(
            _issue(
                "reason_meta_language",
                f"reason contains model/system meta-language: {', '.join(meta)}.",
                "reason",
            )
        )
    if not words.intersection(CONCRETE_TRIGGER_NOUNS):
        issues.append(
            _issue(
                "reason_trigger_noun",
                "reason must name at least one concrete routing trigger noun.",
                "reason",
            )
        )
    return tuple(issues)


def _inspection_issues(
    raw: dict,
    dossier: Dossier,
    inspected_media_ids: frozenset[str] | None,
) -> tuple[ValidationIssue, ...]:
    """Enforce the attachment rule ``prompts.MEDIA_RULES`` states.

    The prompt says a decision describing an attachment the model did not open is
    invalid; this is where that becomes true. ``inspected_media_ids`` of ``None`` means
    the caller is not tracking tool calls at all — a unit test, or a replay — and the
    check stands down rather than rejecting a decision it cannot evaluate.
    """
    if inspected_media_ids is None:
        return ()
    media = dossier.media
    if media.media_type != "image" or media.media_id is None or not media.file_exists:
        return ()
    if media.media_id not in inspected_media_ids:
        return (
            _issue(
                "I7",
                "A readable image attachment must be inspected before a decision is submitted.",
            ),
        )
    if raw.get("media_observation") is None:
        return (
            _issue(
                "I8",
                "An inspected attachment requires a recorded media_observation.",
                "media_observation",
            ),
        )
    return ()


def coerce_and_check(
    raw: dict,
    dossier: Dossier,
    *,
    inspected_media_ids: frozenset[str] | None = None,
) -> ValidationResult:
    """Return a validated decision or a structured, non-raising failure.

    Action and message type use three matching tiers.  Every non-exact repair is logged
    with its original and canonical values and contributes one to ``coercion_count``.
    Evidence ids that the dossier did not offer are dropped before the remaining
    evidence invariants are evaluated.
    """

    schema_issues = _schema_issues(raw)
    if schema_issues:
        return ValidationFailure(stage="schema", issues=schema_issues)

    action, action_tier, action_issue = _coerce_vocabulary(
        "action", raw["action"], ACTIONS
    )
    message_type, type_tier, type_issue = _coerce_vocabulary(
        "message_type", raw["message_type"], MESSAGE_TYPES
    )
    coercion_count = 0
    for field, original, coerced, tier, issue in (
        ("action", raw["action"], action, action_tier, action_issue),
        ("message_type", raw["message_type"], message_type, type_tier, type_issue),
    ):
        if issue is None and coerced is not None and tier not in (None, "exact"):
            _log_coercion(field, original, coerced, tier)
            coercion_count += 1

    coercion_issues = tuple(
        issue for issue in (action_issue, type_issue) if issue is not None
    )
    if coercion_issues:
        return ValidationFailure(
            stage="coercion",
            issues=coercion_issues,
            coercion_count=coercion_count,
        )

    assert action is not None and action_tier is not None
    assert message_type is not None and type_tier is not None

    parsed_evidence = _parse_evidence(raw["evidence_message_ids"])
    confidence = _parse_confidence(raw["confidence"])
    assert parsed_evidence is not None and confidence is not None
    risk_axis = cast(str, raw["risk_axis"])
    relevance_axis = cast(str, raw["relevance_axis"])
    urgency_axis = cast(str, raw["urgency_axis"])

    candidates_by_id = {
        candidate.history_message_id: candidate
        for candidate in dossier.evidence_candidates
    }
    retained_evidence = tuple(
        evidence_id
        for evidence_id in parsed_evidence
        if evidence_id in candidates_by_id
    )
    dropped_evidence = tuple(
        evidence_id
        for evidence_id in parsed_evidence
        if evidence_id not in candidates_by_id
    )
    if dropped_evidence:
        LOGGER.warning(
            "Dropped evidence ids outside this dossier's candidate set: %r",
            dropped_evidence,
            extra={"dropped_evidence_message_ids": dropped_evidence},
        )

    invariant_issues: list[ValidationIssue] = list(
        _inspection_issues(raw, dossier, inspected_media_ids)
    )
    if risk_axis in _RISK_REQUIRES_MUTE and (
        action != "mute" or message_type not in {"scam", "spam"}
    ):
        invariant_issues.append(
            _issue(
                "I1",
                'A scam_or_unsafe risk requires action="mute" and type scam or spam.',
            )
        )
    if action == "notify" and urgency_axis == "none":
        invariant_issues.append(
            _issue("I2", 'action="notify" requires a non-none urgency_axis.')
        )
    if (
        relevance_axis in _UNWANTED_RELEVANCE
        and action == "mute"
        and risk_axis == "clean"
    ):
        has_negative_event = any(
            candidates_by_id[evidence_id].dismissed
            or candidates_by_id[evidence_id].muted_after
            or candidates_by_id[evidence_id].reported
            for evidence_id in retained_evidence
        )
        if not has_negative_event:
            invariant_issues.append(
                _issue(
                    "I3",
                    "A clean unwanted mute requires cited dismissal, mute or report evidence.",
                    "evidence_message_ids",
                )
            )
    # I4 is enforced by construction above: unoffered ids never reach the result.
    if not retained_evidence and candidates_by_id:
        invariant_issues.append(
            _issue(
                "I5",
                'evidence_message_ids may resolve to "none" only when no candidates exist.',
                "evidence_message_ids",
            )
        )
    if len(retained_evidence) > MAX_EVIDENCE_IDS:
        invariant_issues.append(
            _issue(
                "I6",
                f"At most {MAX_EVIDENCE_IDS} evidence ids may be cited.",
                "evidence_message_ids",
            )
        )
    if invariant_issues:
        return ValidationFailure(
            stage="invariants",
            issues=tuple(invariant_issues),
            coercion_count=coercion_count,
            dropped_evidence_message_ids=dropped_evidence,
        )

    reason = cast(str, raw["reason"])
    style_issues = reason_issues(reason)
    if style_issues:
        return ValidationFailure(
            stage="reason_style",
            issues=style_issues,
            coercion_count=coercion_count,
            dropped_evidence_message_ids=dropped_evidence,
        )

    return ValidatedDecision(
        action=cast(Action, action),
        message_type=message_type,
        reason=reason,
        confidence=confidence,
        evidence_message_ids=retained_evidence,
        risk=cast(RiskVerdict, _RISK_AXIS_TO_DECISION[risk_axis]),
        relevance=cast(Relevance, _RELEVANCE_AXIS_TO_DECISION[relevance_axis]),
        urgency=cast(Urgency, urgency_axis),
        media_mismatch=cast(bool, raw.get("media_mismatch", False)),
        media_mismatch_reason=cast(str | None, raw.get("media_mismatch_reason")),
        deadline_minutes=cast(int | None, raw.get("deadline_minutes")),
        material_harm=cast(bool, raw.get("material_harm", False)),
        coercion_count=coercion_count,
    )
