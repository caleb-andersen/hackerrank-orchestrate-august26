"""Precision-grade detection of the two solicitations that justify a hard veto.

Why this exists next to ``context/scanners.py`` rather than inside it
--------------------------------------------------------------------
The repo already runs two tiers of detector, and ``guards/injection.py`` is the
established precedent. ``context/scanners.py`` produces **recall-grade** signals that go
into the dossier for the model to weigh; being slightly over-inclusive there is cheap,
because the model reads the surrounding sentence and can discount a false alarm. The
safety gate cannot. A gate rule overwrites the model's decision without appeal, so its
trigger has to be **precision-grade**, and the two jobs need different thresholds.

That is not a theoretical distinction here. Measured on the 110 participant rows,
``scan_credential_request`` fires on a courier notice whose text is *"no payment or OTP
is required for this delivery"* — the message says the opposite of what the pattern
claims — and ``scan_payment_pressure`` fires on a colleague writing *"the payment worker
retries crossed the alert threshold"*, which is a service name, not a demand. Wiring
either of those into a veto would mute a legitimate business update and a legitimate work
message, and over-muting is the failure mode that costs the most across the scored
columns. So the gate consumes these functions instead, and the dossier keeps the coarser
pair for the model.

What each function recognises
-----------------------------
``asks_for_credential`` requires a **directive aimed at the recipient** whose object is a
secret the recipient holds. A statement that merely mentions a credential never matches,
which is what keeps the courier notice out; and a warning *not* to share one never
matches either, which is what keeps a bank's own security advice out.

``demands_payment`` requires a **demand that the recipient part with money**: an
imperative to pay, a payment asserted as due, a named fee asserted as owing, or a QR/link
payment flow. Mentioning money is not enough. A bare QR reference is not enough either —
this corpus contains a legitimate society notice reading *"please use the society app or
the office QR only"*, and a rule that fired on the word ``QR`` would suppress it.

Both functions return the **matched phrase** rather than a boolean, so the sentence the
gate writes into the ``reason`` cell can quote the specific trigger.

Language coverage
-----------------
English plus romanised Hinglish, because both carry attacks in this corpus and neither
appears in a non-Latin script. The Hinglish patterns are the imperative tails —
``batao``, ``bhejo``, ``karo``, ``kar dena``, ``daal do`` — which follow their object
rather than preceding it, so they need their own direction. French appears here only in
benign traffic and is deliberately not modelled: a pattern that cannot fire cannot be
defended.
"""

import re
import unicodedata
from re import Pattern


# Longer matches are truncated so one pattern cannot flood the reason cell.
_MAX_PHRASE_CHARS = 80
# How far back a negation may sit and still cancel a match.
_NEGATION_LOOKBACK_CHARS = 24

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE = re.compile(r"\s+")

# A directive that cancels the solicitation it precedes: security advice telling the user
# *not* to share a credential, or a notice stating that no payment is required.
_NEGATION = re.compile(
    r"\b(?:never|not|no|dont|don't|do not|nobody|no one|without|avoid|won't|will not)\b"
)

# Verbs that ask the recipient to hand something over.
_TRANSMIT_VERB = (
    r"(?:send|sends|sending|share|shares|sharing|forward|forwards|resend|give|gives"
    r"|provide|provides|submit|submits|enter|enters|type|paste|confirm|confirms"
    r"|verify|verifies|tell|reply\s+with|respond\s+with|revert\s+with)"
)
# The determiner and modifier run that sits between the verb and its object.
_FILLER = r"(?:\s+(?:me|us|us\s+the|the|your|ur|a|this|that|here|back|now|quickly|wallet|bank|account|login|net\s*banking|upi))*"
# Secrets whose name alone identifies them, so a directive plus the noun is enough.
_STRONG_SECRET = (
    r"(?:o\W?t\W?p|one\s*[- ]?\s*time\s+(?:password|pin|code)|otps"
    r"|pin|mpin|m-pin|passcode|password|pass\s*word"
    r"|cvv|card\s+number|debit\s+card\s+(?:number|details)"
    r"|credit\s+card\s+(?:number|details)"
    r"|login\s+code|verification\s+code|security\s+code|access\s+code|auth\s+code)"
)
# Words that are only a secret in a verification context, so they need corroboration.
_WEAK_SECRET = r"(?:code|number|digits)"
# The corroboration a weak object needs before it counts as a credential request.
_VERIFICATION_CONTEXT = re.compile(
    r"\b(?:otp|one\s*time|verification|verify|verified|authenticat\w*|login|log\s*in"
    r"|sign\s*in|account|wallet|kyc|security|\d\s*digit)\b"
)
# Hinglish imperatives, which follow their object instead of preceding it.
_HINGLISH_IMPERATIVE = r"(?:batao|batado|bata\s+do|bhejo|bhej\s+do|bhejna|daal\s+do|daal\s+dena|kar\s+do|kar\s+dena|karo)"

_CREDENTIAL_PATTERNS: tuple[Pattern[str], ...] = (
    # "share your OTP", "confirm your wallet PIN", "reply with the 6 digit login code"
    re.compile(rf"\b{_TRANSMIT_VERB}{_FILLER}(?:\s+\d+\s*[- ]?\s*digit)?\s+{_STRONG_SECRET}\b"),
    # "verification code abhi confirm karo", "OTP abhi batao" — object first.
    re.compile(rf"\b{_STRONG_SECRET}\b.{{0,20}}?\s{_HINGLISH_IMPERATIVE}\b"),
    re.compile(rf"\b{_STRONG_SECRET}\s+(?:abhi\s+|jaldi\s+|now\s+|here\s+)?{_TRANSMIT_VERB}\b"),
)
# Weak-object forms, admitted only when the message also carries verification context.
_WEAK_CREDENTIAL_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(rf"\b{_TRANSMIT_VERB}{_FILLER}(?:\s+\d+\s*[- ]?\s*digit)?\s+{_WEAK_SECRET}\b"),
    re.compile(rf"\b{_WEAK_SECRET}\b.{{0,16}}?\s{_HINGLISH_IMPERATIVE}\b"),
)

# Things a demand asks the recipient to settle.
_PAYABLE = (
    r"(?:fee|fees|amount|amt|charge|charges|due|dues|bill|invoice|penalty|fine|token"
    r"|balance|installment|instalment|emi|maintenance|premium|deposit|clearance"
    r"|rs\.?|inr|₹|\$|\d[\d,.]*)"
)
_PAYMENT_PATTERNS: tuple[Pattern[str], ...] = (
    # "pay the clearance amount", "pay processing fee", "pay Rs 11,000 token"
    re.compile(rf"\bpay\s+(?:the\s+|this\s+|your\s+|a\s+|an\s+)?(?:\w+\s+){{0,2}}{_PAYABLE}"),
    # "pay today", "pay now", "pay before 6 PM"
    re.compile(r"\bpay\b\s+(?:up\s+)?(?:now|today|tonight|immediately|asap|urgently|before|by|within|first)\b"),
    # A QR or link payment flow, in either order.
    re.compile(r"\b(?:scan|open|click|tap|use)\b[^.!?]{0,40}\b(?:qr|link)\b[^.!?]{0,40}\bpay"),
    re.compile(r"\bpay\b[^.!?]{0,40}\b(?:qr\s*code|qr|this\s+link|the\s+link)\b"),
    # A payment asserted as owing rather than requested.
    re.compile(r"\bpayment\s+(?:is\s+)?(?:due|pending|overdue|awaited|required)\b"),
    # A named fee asserted as owing.
    re.compile(
        r"\b(?:processing|clearance|reactivation|activation|customs|handling|release"
        r"|late|service|convenience|registration)\s+(?:fee|fees|charge|charges)\b"
    ),
    # Hinglish: "maintenance payment aaj 5 baje tak kar dena", "paise bhej do"
    re.compile(rf"\b(?:payment|paise|paisa|rupaye|amount)\b.{{0,40}}?\s{_HINGLISH_IMPERATIVE}\b"),
)


def _normalise(text: str) -> str:
    """Fold case and hidden characters while preserving punctuation and digits.

    Digits survive because an amount is part of a demand, and punctuation survives
    because the sentence-bounded patterns use ``.``, ``!`` and ``?`` to stop a match
    running across two unrelated sentences.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", _ZERO_WIDTH.sub("", folded)).strip()


def _is_negated(text: str, start: int) -> bool:
    """Report whether a negation sits close enough in front of a match to cancel it."""
    window = text[max(0, start - _NEGATION_LOOKBACK_CHARS) : start]
    return _NEGATION.search(window) is not None


def _first_match(text: str, patterns: tuple[Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        for match in pattern.finditer(text):
            if _is_negated(text, match.start()):
                continue
            phrase = match.group(0).strip()
            if len(phrase) > _MAX_PHRASE_CHARS:
                return f"{phrase[:_MAX_PHRASE_CHARS]}…"
            return phrase
    return None


def asks_for_credential(text: str) -> str | None:
    """Return the phrase by which ``text`` asks the user for a secret, else ``None``.

    A secret here is an OTP, PIN, password, login or verification code, card number or
    CVV. The match requires a directive aimed at the recipient, so a message that only
    mentions a credential — including one that says none is required — returns ``None``.
    """
    if not text:
        return None
    normalised = _normalise(text)
    if not normalised:
        return None
    strong = _first_match(normalised, _CREDENTIAL_PATTERNS)
    if strong is not None:
        return strong
    if _VERIFICATION_CONTEXT.search(normalised) is None:
        return None
    return _first_match(normalised, _WEAK_CREDENTIAL_PATTERNS)


def demands_payment(text: str) -> str | None:
    """Return the phrase by which ``text`` demands money from the user, else ``None``.

    Covers a direct instruction to pay, a payment or named fee asserted as owing, and a
    QR or link payment flow. Mentioning money, a receipt or a QR code without a demand
    attached returns ``None``.
    """
    if not text:
        return None
    normalised = _normalise(text)
    if not normalised:
        return None
    return _first_match(normalised, _PAYMENT_PATTERNS)
