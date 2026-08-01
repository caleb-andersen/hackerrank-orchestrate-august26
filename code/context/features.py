"""Build a typed, deterministic dossier from loaded participant data."""

import re
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from statistics import median
from typing import Literal, cast

from config import (
    BRAND_MAX_REPORTS,
    BRAND_MIN_AGE_DAYS,
    BRAND_MIN_DOMAIN_AGE_DAYS,
)
from context.index import FeatureIndex, resolve_peer
from context.retrieval import (
    EvidenceCandidate,
    Repetition,
    build_repetition,
    select_evidence,
)
from context.scanners import (
    scan_credential_request,
    scan_injection,
    scan_payment_pressure,
)
from context.text import normalise_text
from context.timewindow import dnd_state, parse_dnd_window
from data.loader import Dataset
from data.schema import (
    BusinessAccount,
    Group,
    GroupMember,
    HistoryMessage,
    Message,
    MessageEvent,
    User,
)


Rate = float | None
_URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|www\.)([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)"
    r"(?::\d+)?(?=[/?#\s),.!;:]|$)"
)


@dataclass(frozen=True, slots=True)
class BrandIntegrity:
    verified: bool
    official_domain: str | None
    domain_used_by_sender: str | None
    domain_mismatch: bool | None
    account_age_days: int
    domain_used_by_sender_age_days: int
    user_reports_30d: int
    verdict: Literal["clean", "suspect", "impersonation"]
    verdict_basis: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SenderIdentity:
    peer_kind: Literal["user", "business"]
    peer_id: str | None
    display_name: str | None
    brand_name: str | None
    category: str | None
    brand_integrity: BrandIntegrity | None


@dataclass(frozen=True, slots=True)
class EngagementRates:
    scope: Literal["user_peer", "global_peer"]
    is_fallback: bool
    basis_note: str
    n: int
    open_rate: Rate
    reply_rate: Rate
    dismiss_rate: Rate
    mute_rate: Rate
    report_rate: Rate
    n_reacted: int
    median_reaction_minutes: float | None


@dataclass(frozen=True, slots=True)
class UserBaseline:
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int
    notifications_sent_30d: int
    notifications_dismissed_total: int
    n_summary_days: int
    baseline_dismiss_rate: Rate
    mean_daily_notifications: float | None


@dataclass(frozen=True, slots=True)
class GroupContext:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    group_messages_30d: int
    user_role: str
    group_muted_by_user: bool
    user_messages_sent_30d: int
    user_messages_read_30d: int
    user_replies_sent_30d: int
    user_notifications_dismissed_30d: int
    group_read_rate: Rate
    group_reply_rate: Rate
    group_dismiss_rate: Rate


@dataclass(frozen=True, slots=True)
class BusinessRelationship:
    why_user_knows_account: str
    last_activity_at: datetime | None
    days_since_last_activity: int | None
    allows_promotions: bool
    promotions_opted_out_at: datetime | None
    opted_out: bool
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: datetime | None
    open_share: Rate


@dataclass(frozen=True, slots=True)
class Relationship:
    peer_engagement: EngagementRates
    peer_global: EngagementRates
    evidence_state: Literal["peer", "global_fallback", "none"]
    user_baseline: UserBaseline
    group_context: GroupContext | None
    business_relationship: BusinessRelationship | None


@dataclass(frozen=True, slots=True)
class ContentSignals:
    raw_text: str
    normalised_text: str
    text_length: int
    is_empty_text: bool
    text_scanned: bool
    forwarded_count: int
    is_forwarded: bool
    url_domains: tuple[str, ...]
    injection_match: str | None
    credential_request: str | None
    payment_pressure: str | None


@dataclass(frozen=True, slots=True)
class Media:
    media_type: Literal["image", "voice"] | None
    media_id: str | None
    file_path: Path | None
    file_exists: bool
    file_size_bytes: int | None
    requires_transcription: bool


@dataclass(frozen=True, slots=True)
class TimingContext:
    created_at: datetime
    local_time: time
    dnd_window_raw: str | None
    dnd_start: time | None
    dnd_end: time | None
    dnd_wraps_midnight: bool
    in_dnd: bool
    minutes_until_dnd_ends: int | None


@dataclass(frozen=True, slots=True)
class Dossier:
    message_id: str
    user_id: str
    conversation_type: Literal["personal", "group", "business"]
    created_at: datetime
    sender_identity: SenderIdentity
    relationship: Relationship
    content_signals: ContentSignals
    repetition: Repetition
    evidence_candidates: tuple[EvidenceCandidate, ...]
    media: Media
    timing: TimingContext


def _rate(numerator: int, denominator: int) -> Rate:
    return None if denominator == 0 else numerator / denominator


def _normalise_domain(domain: str) -> str:
    normalized = domain.strip().casefold()
    return normalized[4:] if normalized.startswith("www.") else normalized


def _domain_mismatch(account: BusinessAccount) -> bool | None:
    if account.official_domain is None or account.domain_used_by_sender is None:
        return None
    official = _normalise_domain(account.official_domain)
    used = _normalise_domain(account.domain_used_by_sender)
    return not (used == official or used.endswith(f".{official}"))


def brand_integrity(account: BusinessAccount) -> BrandIntegrity:
    """Evaluate the specified three-valued business integrity verdict."""
    mismatch = _domain_mismatch(account)
    conditions = (
        mismatch is True,
        not account.verified,
        account.account_age_days < BRAND_MIN_AGE_DAYS,
        account.domain_used_by_sender_age_days < BRAND_MIN_DOMAIN_AGE_DAYS,
        account.user_reports_30d > BRAND_MAX_REPORTS,
    )
    tokens = (
        "domain_mismatch",
        "unverified",
        "account_new",
        "sender_domain_new",
        "reported_by_users",
    )
    fired = tuple(token for token, condition in zip(tokens, conditions) if condition)
    if account.official_domain is None:
        fired = ("official_domain_absent",) + fired

    if all(conditions):
        verdict: Literal["clean", "suspect", "impersonation"] = "impersonation"
    elif (
        mismatch is True
        or (mismatch is None and not account.verified)
        or account.user_reports_30d > BRAND_MAX_REPORTS
    ):
        verdict = "suspect"
    else:
        verdict = "clean"

    return BrandIntegrity(
        verified=account.verified,
        official_domain=account.official_domain,
        domain_used_by_sender=account.domain_used_by_sender,
        domain_mismatch=mismatch,
        account_age_days=account.account_age_days,
        domain_used_by_sender_age_days=account.domain_used_by_sender_age_days,
        user_reports_30d=account.user_reports_30d,
        verdict=verdict,
        verdict_basis=fired,
    )


def _find_user(dataset: Dataset, user_id: str) -> User | None:
    return next((user for user in dataset.users if user.user_id == user_id), None)


def _find_group(dataset: Dataset, group_id: str) -> Group | None:
    return next((group for group in dataset.groups if group.group_id == group_id), None)


def _engagement_rates(
    scope: Literal["user_peer", "global_peer"],
    rows: tuple[HistoryMessage, ...],
    dataset: Dataset,
) -> EngagementRates:
    joined: list[MessageEvent] = []
    for row in rows:
        event = dataset.events_by_user_message.get((row.user_id, row.message_id))
        if event is not None:
            joined.append(event)
    n = len(joined)
    reaction_times = tuple(
        event.reaction_time_minutes
        for event in joined
        if event.reaction_time_minutes is not None
    )
    n_reacted = len(reaction_times)
    median_reaction = (
        None if n_reacted == 0 else float(median(reaction_times))
    )

    if scope == "user_peer":
        basis_note = (
            f"Computed over {n} earlier messages this user received from this sender."
            if n > 0
            else "This user has never received a message from this sender. No per-user rates exist; all rates below are null."
        )
        is_fallback = False
    else:
        basis_note = (
            f"FALLBACK — weaker evidence: this sender's behaviour across all {n} messages it sent to any user, not to this user."
            if n > 0
            else "FALLBACK unavailable: this sender does not appear anywhere in message history."
        )
        is_fallback = True

    return EngagementRates(
        scope=scope,
        is_fallback=is_fallback,
        basis_note=basis_note,
        n=n,
        open_rate=_rate(sum(event.message_opened for event in joined), n),
        reply_rate=_rate(sum(event.message_replied for event in joined), n),
        dismiss_rate=_rate(sum(event.notification_dismissed for event in joined), n),
        mute_rate=_rate(sum(event.muted_after_message for event in joined), n),
        report_rate=_rate(sum(event.message_reported for event in joined), n),
        n_reacted=n_reacted,
        median_reaction_minutes=median_reaction,
    )


def _user_baseline(dataset: Dataset, index: FeatureIndex, user_id: str) -> UserBaseline:
    user = _find_user(dataset, user_id)
    sent, dismissed, n_days = index.daily_totals_by_user.get(user_id, (0, 0, 0))
    return UserBaseline(
        messages_opened_30d=0 if user is None else user.messages_opened_30d,
        messages_replied_30d=0 if user is None else user.messages_replied_30d,
        notifications_dismissed_30d=(
            0 if user is None else user.notifications_dismissed_30d
        ),
        messages_reported_30d=0 if user is None else user.messages_reported_30d,
        notifications_sent_30d=sent,
        notifications_dismissed_total=dismissed,
        n_summary_days=n_days,
        baseline_dismiss_rate=_rate(dismissed, sent),
        mean_daily_notifications=_rate(sent, n_days),
    )


def _group_context(dataset: Dataset, message: Message) -> GroupContext | None:
    if message.conversation_type != "group" or message.group_id is None:
        return None
    group = _find_group(dataset, message.group_id)
    member: GroupMember | None = dataset.group_members_by_group_user.get(
        (message.group_id, message.user_id)
    )
    if group is None or member is None:
        return None
    return GroupContext(
        group_id=group.group_id,
        group_name=group.group_name,
        group_type=group.group_type,
        member_count=group.member_count,
        admin_count=group.admin_count,
        group_messages_30d=group.messages_30d,
        user_role=member.role,
        group_muted_by_user=member.group_muted_by_user,
        user_messages_sent_30d=member.messages_sent_30d,
        user_messages_read_30d=member.messages_read_30d,
        user_replies_sent_30d=member.replies_sent_30d,
        user_notifications_dismissed_30d=member.notifications_dismissed_30d,
        group_read_rate=_rate(member.messages_read_30d, group.messages_30d),
        group_reply_rate=_rate(member.replies_sent_30d, member.messages_read_30d),
        group_dismiss_rate=_rate(
            member.notifications_dismissed_30d,
            group.messages_30d,
        ),
    )


def _business_relationship(
    dataset: Dataset,
    message: Message,
) -> BusinessRelationship | None:
    if message.conversation_type != "business" or message.business_id is None:
        return None
    history = dataset.user_business_by_user_business.get(
        (message.user_id, message.business_id)
    )
    if history is None:
        return None
    denominator = history.messages_opened_30d + history.messages_dismissed_30d
    return BusinessRelationship(
        why_user_knows_account=history.why_user_knows_account,
        last_activity_at=history.last_activity_at,
        days_since_last_activity=(
            None
            if history.last_activity_at is None
            else (message.created_at - history.last_activity_at).days
        ),
        allows_promotions=history.allows_promotions,
        promotions_opted_out_at=history.promotions_opted_out_at,
        opted_out=history.promotions_opted_out_at is not None,
        activity_count_180d=history.activity_count_180d,
        messages_opened_30d=history.messages_opened_30d,
        messages_dismissed_30d=history.messages_dismissed_30d,
        messages_replied_30d=history.messages_replied_30d,
        last_reply_at=history.last_reply_at,
        open_share=_rate(history.messages_opened_30d, denominator),
    )


def _relationship(
    dataset: Dataset,
    index: FeatureIndex,
    message: Message,
    peer_id: str | None,
) -> Relationship:
    if peer_id is None:
        peer_rows: tuple[HistoryMessage, ...] = ()
        global_rows: tuple[HistoryMessage, ...] = ()
    else:
        peer_rows = tuple(
            row
            for row in index.history_by_user_peer.get((message.user_id, peer_id), ())
            if row.created_at < message.created_at
        )
        global_rows = tuple(
            row
            for row in index.history_by_peer.get(peer_id, ())
            if row.created_at < message.created_at
        )
    peer_engagement = _engagement_rates("user_peer", peer_rows, dataset)
    peer_global = _engagement_rates("global_peer", global_rows, dataset)
    if peer_engagement.n > 0:
        evidence_state: Literal["peer", "global_fallback", "none"] = "peer"
    elif peer_global.n > 0:
        evidence_state = "global_fallback"
    else:
        evidence_state = "none"
    return Relationship(
        peer_engagement=peer_engagement,
        peer_global=peer_global,
        evidence_state=evidence_state,
        user_baseline=_user_baseline(dataset, index, message.user_id),
        group_context=_group_context(dataset, message),
        business_relationship=_business_relationship(dataset, message),
    )


def _url_domains(raw_text: str) -> tuple[str, ...]:
    return tuple(
        sorted({_normalise_domain(match.group(1)) for match in _URL_PATTERN.finditer(raw_text)})
    )


def _content_signals(message: Message) -> ContentSignals:
    raw_text = message.message_text
    normalized = normalise_text(raw_text)
    text_length = len(raw_text.strip())
    text_scanned = len(normalized) > 0
    return ContentSignals(
        raw_text=raw_text,
        normalised_text=normalized,
        text_length=text_length,
        is_empty_text=text_length == 0,
        text_scanned=text_scanned,
        forwarded_count=message.forwarded_count,
        is_forwarded=message.forwarded_count > 0,
        url_domains=_url_domains(raw_text),
        injection_match=scan_injection(normalized) if text_scanned else None,
        credential_request=(
            scan_credential_request(normalized) if text_scanned else None
        ),
        payment_pressure=scan_payment_pressure(normalized) if text_scanned else None,
    )


def _sender_identity(dataset: Dataset, message: Message) -> SenderIdentity:
    peer_kind, peer_id = resolve_peer(
        message.conversation_type,
        message.business_id,
        message.sender_user_id,
    )
    account = (
        dataset.business_by_id.get(message.business_id)
        if message.conversation_type == "business" and message.business_id is not None
        else None
    )
    return SenderIdentity(
        peer_kind=peer_kind,
        peer_id=peer_id,
        display_name=None if account is None else account.display_name,
        brand_name=None if account is None else account.brand_name,
        category=None if account is None else account.category,
        brand_integrity=None if account is None else brand_integrity(account),
    )


def _media(dataset: Dataset, message: Message) -> Media:
    media_ref = None
    if message.media_type == "image" and message.media_id is not None:
        media_ref = dataset.images_by_id.get(message.media_id)
    elif message.media_type == "voice" and message.media_id is not None:
        media_ref = dataset.voice_notes_by_id.get(message.media_id)
    file_path = None if media_ref is None else media_ref.file_path
    file_exists = False
    file_size: int | None = None
    if file_path is not None:
        try:
            file_exists = file_path.is_file()
            file_size = file_path.stat().st_size if file_exists else None
        except OSError:
            file_exists = False
            file_size = None
    return Media(
        media_type=cast(Literal["image", "voice"] | None, message.media_type),
        media_id=message.media_id,
        file_path=file_path,
        file_exists=file_exists,
        file_size_bytes=file_size,
        requires_transcription=message.media_type is not None,
    )


def _timing(dataset: Dataset, message: Message) -> TimingContext:
    user = _find_user(dataset, message.user_id)
    raw_window = None if user is None else user.do_not_disturb_window
    window = parse_dnd_window(raw_window)
    in_dnd, minutes = dnd_state(message.created_at, window)
    start = None if window is None else window[0]
    end = None if window is None else window[1]
    return TimingContext(
        created_at=message.created_at,
        local_time=message.created_at.time(),
        dnd_window_raw=raw_window,
        dnd_start=start,
        dnd_end=end,
        dnd_wraps_midnight=(start is not None and end is not None and start > end),
        in_dnd=in_dnd,
        minutes_until_dnd_ends=minutes,
    )


def build_dossier(dataset: Dataset, index: FeatureIndex, message: Message) -> Dossier:
    """Build the complete frozen dossier in the specified dependency order."""
    sender_identity = _sender_identity(dataset, message)
    relationship = _relationship(dataset, index, message, sender_identity.peer_id)
    content_signals = _content_signals(message)
    repetition = build_repetition(message, index, dataset)
    evidence_candidates = select_evidence(message, index, dataset)
    media = _media(dataset, message)
    timing = _timing(dataset, message)
    return Dossier(
        message_id=message.message_id,
        user_id=message.user_id,
        conversation_type=cast(
            Literal["personal", "group", "business"],
            message.conversation_type,
        ),
        created_at=message.created_at,
        sender_identity=sender_identity,
        relationship=relationship,
        content_signals=content_signals,
        repetition=repetition,
        evidence_candidates=evidence_candidates,
        media=media,
        timing=timing,
    )
