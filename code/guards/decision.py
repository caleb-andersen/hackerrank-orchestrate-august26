"""The routing decision as it exists between the model and the CSV.

Two records, one boundary between them.

``ValidatedDecision`` is what the model produced *after* schema validation: the five
contract columns plus the three axes of Decision 1 and the structured media observation.
It is the gate's input and it is never written anywhere.

``FinalDecision`` is what ships. It carries the six contract columns and a ``trace``
whose keys are all underscore-prefixed, which is what keeps the measurement fields —
pre-gate action, pre-gate confidence, which rules fired — out of the graded CSV. The
prefix rule is enforced in ``__post_init__`` rather than documented, so a trace key that
could collide with a contract column cannot be constructed in the first place.

This module holds no logic beyond that invariant. It exists so that the validator, the
gate and the writer share one definition of a decision instead of three.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping


# The three values the ``action`` column may take, ordered elsewhere by severity.
Action = Literal["notify", "digest", "mute"]
# Decision 1's risk axis. ``unsafe`` vetoes; ``suspect`` informs but does not veto.
RiskVerdict = Literal["clean", "suspect", "unsafe"]
# Decision 1's relevance axis, which splits digest from mute.
Relevance = Literal["high", "medium", "low"]
# Decision 1's urgency axis, which splits notify from digest.
Urgency = Literal["immediate", "today", "none"]

# Multiple evidence ids are joined this way in the gold rows.
EVIDENCE_SEPARATOR = ";"
# The contract's stated placeholder when no historical evidence is worth citing.
NO_EVIDENCE = "none"


class GateInvariantError(RuntimeError):
    """Raised when a gate invariant is violated, which is a bug and not bad data."""


@dataclass(frozen=True, slots=True)
class ValidatedDecision:
    """One model-authored routing decision that has already passed schema validation."""

    action: Action
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: tuple[str, ...]
    # Decision 1's three axes, reported separately so the gate can read risk and urgency
    # without re-deriving them from the action the model chose.
    risk: RiskVerdict
    relevance: Relevance
    urgency: Urgency
    # The structured observation from the image or voice tool: whether the attachment
    # actually supports what the text claims, and in what way it does not.
    media_mismatch: bool = False
    media_mismatch_reason: str | None = None
    # Minutes from this message until the window for the action it asks for closes.
    # ``None`` means the message asks for nothing time-bound.
    deadline_minutes: int | None = None
    # True when waiting until the do-not-disturb window ends would cause real harm
    # rather than mere inconvenience.
    material_harm: bool = False
    # Number of action / message-type values repaired by the validator. Keeping this
    # on the validated record makes model schema drift measurable without leaking the
    # measurement into the six-column output.
    coercion_count: int = 0


@dataclass(frozen=True, slots=True)
class FinalDecision:
    """One shipped routing decision plus the measurement trace that never ships."""

    message_id: str
    action: Action
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: tuple[str, ...]
    trace: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        offenders = sorted(key for key in self.trace if not key.startswith("_"))
        if offenders:
            raise GateInvariantError(
                f"Trace keys must be underscore-prefixed so they cannot reach the CSV: "
                f"{offenders}"
            )

    def csv_row(self) -> dict[str, str]:
        """Return exactly the six contract columns, reading nothing from the trace."""
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:.2f}",
            "evidence_message_ids": (
                EVIDENCE_SEPARATOR.join(self.evidence_message_ids)
                if self.evidence_message_ids
                else NO_EVIDENCE
            ),
        }
