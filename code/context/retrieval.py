"""Deterministic ranking of historical repetition and evidence."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from config import (
    BURST_WINDOW_HOURS,
    EVIDENCE_MIN_SCORE,
    EVIDENCE_TOP_K,
    EVENT_RELEVANCE,
    NEAR_DUPLICATE_MIN_JACCARD,
    NEAR_DUPLICATE_TOP_K,
    RECENCY_HALF_LIFE_DAYS,
    W_EVENT,
    W_LEXICAL,
    W_RECENCY,
    W_SAME_GROUP,
    W_SAME_PEER,
)
from context.index import FeatureIndex, resolve_peer
from context.text import jaccard, normalise_text, trigrams
from data.loader import Dataset
from data.schema import HistoryMessage, Message, MessageEvent


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    total: float
    same_peer: float
    same_group: float
    lexical: float
    event_relevance: float
    recency: float


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    history_message_id: str
    score: float
    same_peer: bool
    same_group: bool
    jaccard: float
    event_relevance: float
    recency: float
    created_at: datetime
    days_ago: float
    conversation_type: str
    text_excerpt: str
    opened: bool
    replied: bool
    dismissed: bool
    muted_after: bool
    reported: bool


@dataclass(frozen=True, slots=True)
class NearDuplicate:
    history_message_id: str
    jaccard: float
    created_at: datetime
    days_ago: float
    peer_id: str | None
    same_peer: bool
    opened: bool
    replied: bool
    dismissed: bool
    muted_after: bool
    reported: bool


@dataclass(frozen=True, slots=True)
class Repetition:
    near_duplicate_history: tuple[NearDuplicate, ...]
    max_jaccard: float | None
    duplicate_count_at_threshold: int
    sender_burst_24h: int


def _days_ago(subject_created_at: datetime, row_created_at: datetime) -> float:
    return (subject_created_at - row_created_at).total_seconds() / (24 * 60 * 60)


def _same_group(subject_group_id: str | None, row: HistoryMessage) -> bool:
    return (
        subject_group_id is not None
        and row.conversation_type == "group"
        and row.group_id == subject_group_id
    )


def _event_relevance_term(event: MessageEvent | None) -> float:
    if event is None:
        return 0.0
    terms = (
        EVENT_RELEVANCE["reported"] if event.message_reported else 0.0,
        EVENT_RELEVANCE["muted_after"] if event.muted_after_message else 0.0,
        EVENT_RELEVANCE["replied"] if event.message_replied else 0.0,
        EVENT_RELEVANCE["dismissed"] if event.notification_dismissed else 0.0,
        EVENT_RELEVANCE["opened"] if event.message_opened else 0.0,
    )
    return max(terms)


def _recency_term(subject_created_at: datetime, row_created_at: datetime) -> float:
    days_ago = max(0.0, _days_ago(subject_created_at, row_created_at))
    return 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)


def _event_flags(event: MessageEvent | None) -> tuple[bool, bool, bool, bool, bool]:
    if event is None:
        return (False, False, False, False, False)
    return (
        event.message_opened,
        event.message_replied,
        event.notification_dismissed,
        event.muted_after_message,
        event.message_reported,
    )


def score_evidence(
    subject_peer_id: str | None,
    subject_group_id: str | None,
    subject_trigrams: frozenset[str],
    subject_created_at: datetime,
    row: HistoryMessage,
    row_peer_id: str | None,
    row_trigrams: frozenset[str],
    event: MessageEvent | None,
) -> EvidenceScore:
    """Score one historical row and retain every normalized score term."""
    same_peer = float(
        subject_peer_id is not None and row_peer_id == subject_peer_id
    )
    same_group = float(_same_group(subject_group_id, row))
    similarity = jaccard(subject_trigrams, row_trigrams)
    lexical = 0.0 if similarity is None else similarity
    event_relevance = _event_relevance_term(event)
    recency = _recency_term(subject_created_at, row.created_at)
    total = (
        W_SAME_PEER * same_peer
        + W_LEXICAL * lexical
        + W_SAME_GROUP * same_group
        + W_EVENT * event_relevance
        + W_RECENCY * recency
    )
    return EvidenceScore(
        total=total,
        same_peer=same_peer,
        same_group=same_group,
        lexical=lexical,
        event_relevance=event_relevance,
        recency=recency,
    )


def _sort_evidence(
    candidates: list[EvidenceCandidate],
) -> tuple[EvidenceCandidate, ...]:
    ordered = sorted(candidates, key=lambda candidate: candidate.history_message_id)
    ordered = sorted(ordered, key=lambda candidate: candidate.created_at, reverse=True)
    return tuple(sorted(ordered, key=lambda candidate: candidate.score, reverse=True))


def select_evidence(
    message: Message,
    index: FeatureIndex,
    dataset: Dataset,
    k: int = EVIDENCE_TOP_K,
) -> tuple[EvidenceCandidate, ...]:
    """Select the strongest earlier history rows for the current user."""
    _peer_kind, subject_peer_id = resolve_peer(
        message.conversation_type,
        message.business_id,
        message.sender_user_id,
    )
    subject_group_id = message.group_id if message.conversation_type == "group" else None
    subject_trigrams = trigrams(normalise_text(message.message_text))
    candidates: list[EvidenceCandidate] = []

    for row in dataset.history_by_user.get(message.user_id, ()):
        if row.created_at >= message.created_at:
            continue
        _row_peer_kind, row_peer_id = resolve_peer(
            row.conversation_type,
            row.business_id,
            row.sender_user_id,
        )
        row_trigrams = index.trigrams_by_history_id.get(row.message_id)
        if row_trigrams is None:
            row_trigrams = trigrams(normalise_text(row.message_text))
        event = dataset.events_by_user_message.get((row.user_id, row.message_id))
        scored = score_evidence(
            subject_peer_id=subject_peer_id,
            subject_group_id=subject_group_id,
            subject_trigrams=subject_trigrams,
            subject_created_at=message.created_at,
            row=row,
            row_peer_id=row_peer_id,
            row_trigrams=row_trigrams,
            event=event,
        )
        if scored.total < EVIDENCE_MIN_SCORE:
            continue
        opened, replied, dismissed, muted_after, reported = _event_flags(event)
        candidates.append(
            EvidenceCandidate(
                history_message_id=row.message_id,
                score=scored.total,
                same_peer=bool(scored.same_peer),
                same_group=bool(scored.same_group),
                jaccard=scored.lexical,
                event_relevance=scored.event_relevance,
                recency=scored.recency,
                created_at=row.created_at,
                days_ago=_days_ago(message.created_at, row.created_at),
                conversation_type=row.conversation_type,
                text_excerpt=normalise_text(row.message_text)[:160],
                opened=opened,
                replied=replied,
                dismissed=dismissed,
                muted_after=muted_after,
                reported=reported,
            )
        )

    return _sort_evidence(candidates)[: max(0, k)]


def _sort_near_duplicates(
    matches: list[NearDuplicate],
) -> tuple[NearDuplicate, ...]:
    ordered = sorted(matches, key=lambda match: match.history_message_id)
    ordered = sorted(ordered, key=lambda match: match.created_at, reverse=True)
    return tuple(sorted(ordered, key=lambda match: match.jaccard, reverse=True))


def _sender_burst(
    message: Message,
    subject_peer_id: str | None,
    rows: tuple[HistoryMessage, ...] | list[HistoryMessage],
) -> int:
    if subject_peer_id is None:
        return 0
    start = message.created_at - timedelta(hours=BURST_WINDOW_HOURS)
    count = 0
    for row in rows:
        if not start <= row.created_at < message.created_at:
            continue
        _row_peer_kind, row_peer_id = resolve_peer(
            row.conversation_type,
            row.business_id,
            row.sender_user_id,
        )
        if row_peer_id == subject_peer_id:
            count += 1
    return count


def build_repetition(
    message: Message,
    index: FeatureIndex,
    dataset: Dataset,
    k: int = NEAR_DUPLICATE_TOP_K,
) -> Repetition:
    """Measure near-duplicate text and same-sender burst behavior."""
    _peer_kind, subject_peer_id = resolve_peer(
        message.conversation_type,
        message.business_id,
        message.sender_user_id,
    )
    subject_trigrams = trigrams(normalise_text(message.message_text))
    user_history = dataset.history_by_user.get(message.user_id, ())
    comparable: list[float] = []
    matches: list[NearDuplicate] = []

    for row in user_history:
        if row.created_at >= message.created_at:
            continue
        row_trigrams = index.trigrams_by_history_id.get(row.message_id)
        if row_trigrams is None:
            row_trigrams = trigrams(normalise_text(row.message_text))
        similarity = jaccard(subject_trigrams, row_trigrams)
        if similarity is None:
            continue
        comparable.append(similarity)
        if similarity < NEAR_DUPLICATE_MIN_JACCARD:
            continue
        _row_peer_kind, row_peer_id = resolve_peer(
            row.conversation_type,
            row.business_id,
            row.sender_user_id,
        )
        event = dataset.events_by_user_message.get((row.user_id, row.message_id))
        opened, replied, dismissed, muted_after, reported = _event_flags(event)
        matches.append(
            NearDuplicate(
                history_message_id=row.message_id,
                jaccard=similarity,
                created_at=row.created_at,
                days_ago=_days_ago(message.created_at, row.created_at),
                peer_id=row_peer_id,
                same_peer=(subject_peer_id is not None and row_peer_id == subject_peer_id),
                opened=opened,
                replied=replied,
                dismissed=dismissed,
                muted_after=muted_after,
                reported=reported,
            )
        )

    sorted_matches = _sort_near_duplicates(matches)
    return Repetition(
        near_duplicate_history=sorted_matches[: max(0, k)],
        max_jaccard=max(comparable) if comparable else None,
        duplicate_count_at_threshold=len(matches),
        sender_burst_24h=_sender_burst(message, subject_peer_id, user_history),
    )
