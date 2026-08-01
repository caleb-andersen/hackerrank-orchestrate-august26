"""Typed, immutable records for the participant-facing dataset."""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


ACTIONS: tuple[str, ...] = ("notify", "digest", "mute")
MESSAGE_TYPES: tuple[str, ...] = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)


@dataclass(frozen=True, slots=True)
class Message:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: str | None
    business_id: str | None
    sender_user_id: str | None
    created_at: datetime
    message_text: str
    media_type: str | None
    media_id: str | None
    forwarded_count: int


@dataclass(frozen=True, slots=True)
class User:
    user_id: str
    do_not_disturb_window: str | None
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


@dataclass(frozen=True, slots=True)
class Group:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: datetime
    messages_30d: int


@dataclass(frozen=True, slots=True)
class GroupMember:
    group_id: str
    user_id: str
    role: str
    joined_at: datetime
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool


@dataclass(frozen=True, slots=True)
class BusinessAccount:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str | None
    domain_used_by_sender: str | None
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int


@dataclass(frozen=True, slots=True)
class UserBusinessHistory:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: datetime | None
    allows_promotions: bool
    promotions_opted_out_at: datetime | None
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: datetime | None


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    message_id: str
    user_id: str
    conversation_type: str
    group_id: str | None
    business_id: str | None
    sender_user_id: str | None
    created_at: datetime
    message_text: str
    media_type: str | None
    media_id: str | None
    forwarded_count: int


@dataclass(frozen=True, slots=True)
class MessageEvent:
    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: float | None
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool


@dataclass(frozen=True, slots=True)
class DailyNotificationSummary:
    user_id: str
    date: date
    notifications_sent: int
    notifications_dismissed: int


@dataclass(frozen=True, slots=True)
class MediaRef:
    media_id: str
    file_path: Path
