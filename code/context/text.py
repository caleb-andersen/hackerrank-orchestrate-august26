"""Shared deterministic text normalization and similarity helpers."""

import unicodedata

from config import TRIGRAM_N


def normalise_text(text: str) -> str:
    """Normalize text while preserving Unicode letters and digits."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters = (
        character if unicodedata.category(character)[0] in {"L", "N"} else " "
        for character in normalized
    )
    return " ".join("".join(characters).split())


def trigrams(text: str) -> frozenset[str]:
    """Return unpadded character trigrams from already-normalized text."""
    return frozenset(
        text[index : index + TRIGRAM_N]
        for index in range(len(text) - TRIGRAM_N + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float | None:
    """Return set overlap, or None when either side is not comparable."""
    if not left or not right:
        return None
    return len(left & right) / len(left | right)
