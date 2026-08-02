"""Author a reason sentence in code, from dossier facts, inside the style contract.

Two callers write reason cells without a model: the safety gate, when a rule fires, and
the failure policy in ``agent.loop``, when a decision is sound but its sentence is not.
Both have to satisfy the same contract ``validate.reason_issues`` enforces on the model —
one sentence, 60–160 characters, third person, no meta-language, at least one concrete
trigger noun — so the formatting helpers live here once rather than twice.

What this module will not do
----------------------------
Every sentence it returns is a statement about the *message*, drawn from a fact the
dossier already computed. None of them describes the pipeline, and none of them asserts a
failure that did not happen. That is the whole point of the module: a reason cell is
graded, and a graded cell that misdescribes its own row is worse than a vague one.

``repair`` is reached only after a decision has passed schema, vocabulary and every
cross-field invariant, so the action, message type, evidence and axes it is paired with
are known-good. It replaces prose, never judgement.
"""

from __future__ import annotations

from context.features import Dossier
from guards.decision import ValidatedDecision
from guards.injection import looks_like_injection
from guards.solicitation import asks_for_credential, demands_payment
from guards.validate import reason_issues


def percent(rate: float) -> str:
    return f"{round(rate * 100)}%"


def plural(count: int, noun: str) -> str:
    """Agree the noun with its count, because the reason cell is a graded column."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _quote(phrase: str) -> str:
    """Strip the quote characters that would terminate the quoted span in the sentence."""
    return phrase.replace('"', "").replace("“", "").replace("”", "").strip()


def _humanise(token: str) -> str:
    """Render a snake_case dataset value as prose without inventing anything."""
    return token.replace("_", " ").strip()


def _phrase_signals(dossier: Dossier) -> tuple[str | None, str | None, str | None]:
    """Re-run the *precision* detectors, not the coarse dossier scanners.

    ``ContentSignals`` carries a cheap substring scan used for retrieval and prompt
    facts; the gate consumes ``guards.solicitation`` and ``guards.injection`` instead,
    which are narrower. The two disagree, and on this corpus the disagreement is not
    theoretical: a FedEx notice reading "no payment or OTP is required for this delivery"
    sets the coarse ``credential_request`` signal to "otp is" while the precision detector
    correctly returns nothing.

    Reading the coarse signal here put that false claim in a graded reason cell, asserting
    a credential request on a message that explicitly disclaims one. Reading the same
    detectors the gate reads means the repaired sentence and the gate's own sentence can
    never contradict each other about the same row.
    """
    text = dossier.content_signals.normalised_text
    if not text:
        return None, None, None
    return (
        asks_for_credential(text),
        looks_like_injection(text),
        demands_payment(text),
    )


def _adverse(dossier: Dossier, decision: ValidatedDecision) -> list[str]:
    """Facts that argue for suppressing this message, most decisive first."""
    identity = dossier.sender_identity
    peer = dossier.relationship.peer_engagement
    integrity = identity.brand_integrity
    business = dossier.relationship.business_relationship
    repetition = dossier.repetition
    signals = dossier.content_signals
    credential, injection, payment = _phrase_signals(dossier)

    out: list[str] = []
    if credential:
        out.append(
            f'A request for a credential worded "{_quote(credential)}" '
            f"decided this message ahead of any standing the sender has."
        )
    if injection:
        # "delivery" and "sender" are load-bearing, not decoration: the contract requires
        # a concrete trigger noun, and a sentence about instruction-shaped text otherwise
        # contains none, so it would silently lose to the next candidate.
        out.append(
            f'Text trying to dictate its own delivery with "{_quote(injection)}" '
            f"was recorded as evidence against this sender."
        )
    if payment:
        out.append(
            f'A demand for payment worded "{_quote(payment)}" decided '
            f"this message against the sender's history."
        )
    if integrity is not None and integrity.verdict == "impersonation":
        brand = identity.brand_name or "this business"
        out.append(
            f"A sender domain that does not match the official {brand} domain, with "
            f"{plural(integrity.user_reports_30d, 'user report')}, decided this notice."
        )
    if business is not None and business.opted_out and decision.message_type == "promotion":
        out.append(
            "A promotion from a business whose promotions this user has already opted "
            "out of decided this message."
        )
    if peer.dismiss_rate is not None and peer.n and peer.dismiss_rate >= 0.5:
        out.append(
            f"This recipient dismissed {percent(peer.dismiss_rate)} of the "
            f"{plural(peer.n, 'previous message')} from this sender."
        )
    if repetition.duplicate_count_at_threshold:
        out.append(
            f"This message repeats "
            f"{plural(repetition.duplicate_count_at_threshold, 'earlier message')} "
            f"from the same sender in the retained history."
        )
    if signals.is_forwarded:
        out.append(
            f"This message arrived as a forward with a hop count of "
            f"{signals.forwarded_count}, which decided it against the sender's history."
        )
    return out


def _supporting(dossier: Dossier, decision: ValidatedDecision) -> list[str]:
    """Facts that argue for delivering this message, most decisive first."""
    relationship = dossier.relationship
    peer = relationship.peer_engagement
    business = relationship.business_relationship
    group = relationship.group_context

    out: list[str] = []
    if decision.deadline_minutes is not None and peer.open_rate is not None and peer.n:
        out.append(
            f"The message states a deadline and this recipient opens "
            f"{percent(peer.open_rate)} of the {plural(peer.n, 'previous message')} "
            f"from this sender."
        )
    if group is not None and group.sender_role == "admin":
        opens = (
            f" and this recipient opens {percent(peer.open_rate)} of them"
            if peer.open_rate is not None and peer.n
            else ""
        )
        out.append(f"A group admin in {group.group_name} sent this notice{opens}.")
    if business is not None and business.why_user_knows_account:
        rate = (
            f" opened {percent(peer.open_rate)} of the time"
            if peer.open_rate is not None and peer.n
            else ""
        )
        out.append(
            f"A business notice matching this recipient's "
            f"{_humanise(business.why_user_knows_account)} decided this message, from an "
            f"account{rate}."
        )
    if peer.open_rate is not None and peer.n:
        out.append(
            f"This recipient opens {percent(peer.open_rate)} of the "
            f"{plural(peer.n, 'previous message')} from this sender."
        )
    return out


def _candidates(dossier: Dossier, decision: ValidatedDecision) -> list[str]:
    """Order the true things that could be said about this row, most decisive first.

    Both groups are true of the row; which one *decided* it depends on which way the row
    came out. Naming a suppressive fact under a ``notify`` — "this message repeats 2
    earlier messages" on a delivery update that was delivered — is a true sentence that
    explains the opposite of what happened, and the reason column is graded on whether it
    explains the action beside it.

    So a ``mute`` leads with the adverse facts and a ``notify`` leads with the supporting
    ones. ``digest`` leads with the adverse group too: a deferral is a partial
    suppression, and what makes a row wait rather than interrupt is nearly always the
    weaker half of its engagement history.

    The exception is the phrase-bearing content signals at the head of ``_adverse``. A
    credential request or an injection attempt outranks the direction of the decision
    entirely, because the safety gate is about to overwrite this sentence on exactly those
    rows and the two should already agree.
    """
    adverse = _adverse(dossier, decision)
    supporting = _supporting(dossier, decision)
    has_phrase_signal = any(_phrase_signals(dossier))

    if decision.action == "notify" and not has_phrase_signal:
        ordered = supporting + adverse
    else:
        ordered = adverse + supporting

    # The floor. True on any row that reaches it: the groups above are exhaustive over
    # every dossier fact that could have carried a more specific sentence, so arriving
    # here means the row had no retained peer history and no fired content signal.
    ordered.append(
        "No earlier history from this sender was retained for this recipient, so this "
        "message was decided on its own content."
    )
    return ordered


def repair(dossier: Dossier, decision: ValidatedDecision) -> str:
    """Return the most decisive true sentence about this row that passes the contract.

    Candidates are tried in order and the first one that satisfies ``reason_issues`` is
    returned. A candidate can fail on length alone — a long group name or a long quoted
    phrase overruns 160 characters — and dropping to the next fact is the right response,
    because the next fact is also true. The floor sentence is checked like the rest; if
    even that fails the caller still gets it, and the caller is responsible for the
    contract from there.
    """
    candidates = _candidates(dossier, decision)
    for candidate in candidates:
        if not reason_issues(candidate):
            return candidate
    return candidates[-1]
