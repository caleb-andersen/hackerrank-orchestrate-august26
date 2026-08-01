"""Precomputed joins for deterministic dossier construction."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from context.text import normalise_text, trigrams
from data.loader import Dataset
from data.schema import HistoryMessage


@dataclass(frozen=True, slots=True)
class FeatureIndex:
    history_by_user_peer: dict[tuple[str, str], tuple[HistoryMessage, ...]]
    history_by_peer: dict[str, tuple[HistoryMessage, ...]]
    trigrams_by_history_id: dict[str, frozenset[str]]
    daily_totals_by_user: dict[str, tuple[int, int, int]]


def resolve_peer(
    conversation_type: str,
    business_id: str | None,
    sender_user_id: str | None,
) -> tuple[Literal["user", "business"], str | None]:
    """Resolve the sender identity consistently for current and history rows."""
    if conversation_type == "business":
        return ("business", business_id or None)
    return ("user", sender_user_id or None)


def build_feature_index(dataset: Dataset) -> FeatureIndex:
    """Build immutable row collections and precomputed text features."""
    by_user_peer: defaultdict[tuple[str, str], list[HistoryMessage]] = defaultdict(list)
    by_peer: defaultdict[str, list[HistoryMessage]] = defaultdict(list)
    trigram_index: dict[str, frozenset[str]] = {}

    for row in dataset.message_history:
        _peer_kind, peer_id = resolve_peer(
            row.conversation_type,
            row.business_id,
            row.sender_user_id,
        )
        trigram_index[row.message_id] = trigrams(normalise_text(row.message_text))
        if peer_id is not None:
            by_user_peer[(row.user_id, peer_id)].append(row)
            by_peer[peer_id].append(row)

    sort_key = lambda row: (row.created_at, row.message_id)
    history_by_user_peer = {
        key: tuple(sorted(rows, key=sort_key))
        for key, rows in by_user_peer.items()
    }
    history_by_peer = {
        key: tuple(sorted(rows, key=sort_key))
        for key, rows in by_peer.items()
    }

    daily_build: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for summary in dataset.daily_notification_summary:
        totals = daily_build[summary.user_id]
        totals[0] += summary.notifications_sent
        totals[1] += summary.notifications_dismissed
        totals[2] += 1
    daily_totals = {
        user_id: (totals[0], totals[1], totals[2])
        for user_id, totals in daily_build.items()
    }

    return FeatureIndex(
        history_by_user_peer=history_by_user_peer,
        history_by_peer=history_by_peer,
        trigrams_by_history_id=trigram_index,
        daily_totals_by_user=daily_totals,
    )
