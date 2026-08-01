"""Pure scanners for adversarial or safety-relevant message text."""

import re
from re import Pattern


_INJECTION_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    (re.compile(r"\b(?:ignore|disregard|forget) (?:all )?(?:previous|prior|above) (?:rules|instructions)\b"), "ignore-prior-rules"),
    (re.compile(r"\b(?:system|operator|administrator) (?:message|instruction|override|notice)\b"), "asserted-system-voice"),
    (re.compile(r"\b(?:always|must|should) (?:notify|digest|mute|suppress)\b"), "asserted-routing-action"),
    (re.compile(r"\b(?:i am|this is) (?:a )?(?:verified|trusted|official) (?:sender|account|business)\b"), "asserted-sender-metadata"),
)

_CREDENTIAL_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    (re.compile(r"\b(?:send|share|enter|provide|confirm) (?:your )?(?:one time password|otp|pin|password|cvv)\b"), "credential-solicitation"),
    (re.compile(r"\b(?:kyc|identity|account) (?:re ?verification|re ?activation|required verification)\b"), "account-reverification"),
    (re.compile(r"\b(?:otp|pin|password|cvv) (?:is|required|needed|code)\b"), "credential-request"),
)

_PAYMENT_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    (re.compile(r"\b(?:pay|payment|transfer|settle)\b.{0,60}\b(?:immediately|urgent|today|now|expires?|deadline|penalty|late fee|suspend)\b"), "urgent-payment-demand"),
    (re.compile(r"\b(?:immediately|urgent|today|now|expires?|deadline|penalty|late fee|suspend)\b.{0,60}\b(?:pay|payment|transfer|settle)\b"), "urgent-payment-demand"),
)


def _scan(text: str, patterns: tuple[tuple[Pattern[str], str], ...]) -> str | None:
    if not text:
        return None
    for pattern, _label in patterns:
        match = pattern.search(text)
        if match is not None:
            phrase = match.group(0)
            return phrase if len(phrase) <= 80 else f"{phrase[:80]}…"
    return None


def scan_injection(text: str) -> str | None:
    """Return the first matched router-directed instruction phrase."""
    return _scan(text, _INJECTION_PATTERNS)


def scan_credential_request(text: str) -> str | None:
    """Return the first matched credential solicitation phrase."""
    return _scan(text, _CREDENTIAL_PATTERNS)


def scan_payment_pressure(text: str) -> str | None:
    """Return the first matched urgent payment phrase."""
    return _scan(text, _PAYMENT_PATTERNS)
