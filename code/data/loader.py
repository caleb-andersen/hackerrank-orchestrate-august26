"""Load the twelve non-label dataset CSVs into typed records and indexes."""

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, TypeVar

from data.schema import (
    BusinessAccount,
    DailyNotificationSummary,
    Group,
    GroupMember,
    HistoryMessage,
    MediaRef,
    Message,
    MessageEvent,
    User,
    UserBusinessHistory,
)


@dataclass(frozen=True, slots=True)
class Dataset:
    messages: list[Message]
    output_row_order: list[str]
    users: list[User]
    groups: list[Group]
    group_members: list[GroupMember]
    business_accounts: list[BusinessAccount]
    user_business_history: list[UserBusinessHistory]
    message_history: list[HistoryMessage]
    message_events: list[MessageEvent]
    images: list[MediaRef]
    voice_notes: list[MediaRef]
    daily_notification_summary: list[DailyNotificationSummary]
    history_by_user: dict[str, list[HistoryMessage]]
    events_by_user_message: dict[tuple[str, str], MessageEvent]
    group_members_by_group_user: dict[tuple[str, str], GroupMember]
    business_by_id: dict[str, BusinessAccount]
    user_business_by_user_business: dict[tuple[str, str], UserBusinessHistory]
    images_by_id: dict[str, MediaRef]
    voice_notes_by_id: dict[str, MediaRef]
    daily_summary_by_user: dict[str, list[DailyNotificationSummary]]


_Record = TypeVar("_Record")
_Key = TypeVar("_Key")


def _read_csv(path: Path, required_headers: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        normalized_headers: dict[str, str] = {}
        for original in reader.fieldnames:
            normalized = original.lstrip("\ufeff").strip().casefold()
            if not normalized:
                raise ValueError(f"CSV contains an empty header: {path}")
            if normalized in normalized_headers:
                raise ValueError(f"CSV contains duplicate header {normalized!r}: {path}")
            normalized_headers[normalized] = original

        missing = [header for header in required_headers if header not in normalized_headers]
        if missing:
            raise ValueError(f"CSV is missing headers {missing}: {path}")

        rows: list[dict[str, str]] = []
        try:
            for row in reader:
                if None in row:
                    raise ValueError(f"CSV row has more fields than headers at line {reader.line_num}: {path}")
                normalized_row: dict[str, str] = {}
                for normalized, original in normalized_headers.items():
                    value = row.get(original)
                    normalized_row[normalized] = "" if value is None else value
                rows.append(normalized_row)
        except csv.Error as error:
            raise ValueError(f"Malformed CSV near line {reader.line_num}: {path}: {error}") from error
    return rows


def _required(row: dict[str, str], field: str, path: Path) -> str:
    value = row[field].strip()
    if not value:
        raise ValueError(f"Required field {field!r} is empty in {path}")
    return value


def _optional(row: dict[str, str], field: str) -> str | None:
    value = row[field].strip()
    return value or None


def _integer(row: dict[str, str], field: str, path: Path) -> int:
    value = _required(row, field, path)
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Field {field!r} must be an integer in {path}") from error


def _optional_float(row: dict[str, str], field: str, path: Path) -> float | None:
    value = _optional(row, field)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"Field {field!r} must be a number in {path}") from error


def _boolean(row: dict[str, str], field: str, path: Path) -> bool:
    value = _required(row, field, path).casefold()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Field {field!r} must be a boolean in {path}")


def _datetime(row: dict[str, str], field: str, path: Path) -> datetime:
    value = _required(row, field, path)
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Field {field!r} must be an ISO-8601 timestamp in {path}") from error


def _optional_datetime(row: dict[str, str], field: str, path: Path) -> datetime | None:
    value = _optional(row, field)
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Field {field!r} must be an ISO-8601 timestamp in {path}") from error


def _date(row: dict[str, str], field: str, path: Path) -> date:
    value = _required(row, field, path)
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Field {field!r} must be an ISO-8601 date in {path}") from error


def _media_path(raw_path: str, dataset_dir: Path, source_path: Path) -> Path:
    relative = Path(raw_path)
    candidates = [relative] if relative.is_absolute() else [dataset_dir / relative, dataset_dir.parent / relative]
    dataset_root = dataset_dir.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(dataset_root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    raise ValueError(f"Media path must resolve to a file inside the dataset directory: {source_path}")


def _media_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_index(records: list[_Record], key: Callable[[_Record], _Key], name: str) -> dict[_Key, _Record]:
    index: dict[_Key, _Record] = {}
    for record in records:
        record_key = key(record)
        if record_key in index:
            raise ValueError(f"Duplicate key in {name}: {record_key!r}")
        index[record_key] = record
    return index


def _load_messages(path: Path, record_type: type[Message] | type[HistoryMessage]) -> list[Message] | list[HistoryMessage]:
    headers = (
        "message_id",
        "user_id",
        "conversation_type",
        "group_id",
        "business_id",
        "sender_user_id",
        "created_at",
        "message_text",
        "media_type",
        "media_id",
        "forwarded_count",
    )
    records: list[Message] | list[HistoryMessage] = []
    for row in _read_csv(path, headers):
        record = record_type(
            message_id=_required(row, "message_id", path),
            user_id=_required(row, "user_id", path),
            conversation_type=_required(row, "conversation_type", path),
            group_id=_optional(row, "group_id"),
            business_id=_optional(row, "business_id"),
            sender_user_id=_optional(row, "sender_user_id"),
            created_at=_datetime(row, "created_at", path),
            message_text=row["message_text"],
            media_type=_optional(row, "media_type"),
            media_id=_optional(row, "media_id"),
            forwarded_count=_integer(row, "forwarded_count", path),
        )
        records.append(record)
    return records


def load_dataset(dataset_dir: str | Path) -> Dataset:
    """Load participant inputs without opening the labeled sample dataset."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    messages_path = root / "messages.csv"
    output_path = root / "output.csv"
    users_path = root / "users.csv"
    groups_path = root / "groups.csv"
    group_members_path = root / "group_members.csv"
    businesses_path = root / "business_accounts.csv"
    user_business_path = root / "user_business_history.csv"
    history_path = root / "message_history.csv"
    events_path = root / "message_events.csv"
    images_path = root / "images.csv"
    voice_notes_path = root / "voice_notes.csv"
    daily_summary_path = root / "daily_notification_summary.csv"

    loaded_messages = _load_messages(messages_path, Message)
    messages = [record for record in loaded_messages if isinstance(record, Message)]
    output_row_order = [
        _required(row, "message_id", output_path)
        for row in _read_csv(output_path, ("message_id",))
    ]

    users = [
        User(
            user_id=_required(row, "user_id", users_path),
            do_not_disturb_window=_optional(row, "do_not_disturb_window"),
            messages_opened_30d=_integer(row, "messages_opened_30d", users_path),
            messages_replied_30d=_integer(row, "messages_replied_30d", users_path),
            notifications_dismissed_30d=_integer(row, "notifications_dismissed_30d", users_path),
            messages_reported_30d=_integer(row, "messages_reported_30d", users_path),
        )
        for row in _read_csv(
            users_path,
            (
                "user_id",
                "do_not_disturb_window",
                "messages_opened_30d",
                "messages_replied_30d",
                "notifications_dismissed_30d",
                "messages_reported_30d",
            ),
        )
    ]

    groups = [
        Group(
            group_id=_required(row, "group_id", groups_path),
            group_name=_required(row, "group_name", groups_path),
            group_type=_required(row, "group_type", groups_path),
            member_count=_integer(row, "member_count", groups_path),
            admin_count=_integer(row, "admin_count", groups_path),
            created_at=_datetime(row, "created_at", groups_path),
            messages_30d=_integer(row, "messages_30d", groups_path),
        )
        for row in _read_csv(
            groups_path,
            ("group_id", "group_name", "group_type", "member_count", "admin_count", "created_at", "messages_30d"),
        )
    ]

    group_members = [
        GroupMember(
            group_id=_required(row, "group_id", group_members_path),
            user_id=_required(row, "user_id", group_members_path),
            role=_required(row, "role", group_members_path),
            joined_at=_datetime(row, "joined_at", group_members_path),
            messages_sent_30d=_integer(row, "messages_sent_30d", group_members_path),
            messages_read_30d=_integer(row, "messages_read_30d", group_members_path),
            replies_sent_30d=_integer(row, "replies_sent_30d", group_members_path),
            notifications_dismissed_30d=_integer(row, "notifications_dismissed_30d", group_members_path),
            group_muted_by_user=_boolean(row, "group_muted_by_user", group_members_path),
        )
        for row in _read_csv(
            group_members_path,
            (
                "group_id",
                "user_id",
                "role",
                "joined_at",
                "messages_sent_30d",
                "messages_read_30d",
                "replies_sent_30d",
                "notifications_dismissed_30d",
                "group_muted_by_user",
            ),
        )
    ]

    business_accounts = [
        BusinessAccount(
            business_id=_required(row, "business_id", businesses_path),
            display_name=_required(row, "display_name", businesses_path),
            brand_name=_required(row, "brand_name", businesses_path),
            category=_required(row, "category", businesses_path),
            verified=_boolean(row, "verified", businesses_path),
            official_domain=_optional(row, "official_domain"),
            domain_used_by_sender=_optional(row, "domain_used_by_sender"),
            account_age_days=_integer(row, "account_age_days", businesses_path),
            messages_sent_30d=_integer(row, "messages_sent_30d", businesses_path),
            user_reports_30d=_integer(row, "user_reports_30d", businesses_path),
            domain_used_by_sender_age_days=_integer(row, "domain_used_by_sender_age_days", businesses_path),
        )
        for row in _read_csv(
            businesses_path,
            (
                "business_id",
                "display_name",
                "brand_name",
                "category",
                "verified",
                "official_domain",
                "domain_used_by_sender",
                "account_age_days",
                "messages_sent_30d",
                "user_reports_30d",
                "domain_used_by_sender_age_days",
            ),
        )
    ]

    user_business_history = [
        UserBusinessHistory(
            user_id=_required(row, "user_id", user_business_path),
            business_id=_required(row, "business_id", user_business_path),
            why_user_knows_account=_required(row, "why_user_knows_account", user_business_path),
            last_activity_at=_optional_datetime(row, "last_activity_at", user_business_path),
            allows_promotions=_boolean(row, "allows_promotions", user_business_path),
            promotions_opted_out_at=_optional_datetime(row, "promotions_opted_out_at", user_business_path),
            activity_count_180d=_integer(row, "activity_count_180d", user_business_path),
            messages_opened_30d=_integer(row, "messages_opened_30d", user_business_path),
            messages_dismissed_30d=_integer(row, "messages_dismissed_30d", user_business_path),
            messages_replied_30d=_integer(row, "messages_replied_30d", user_business_path),
            last_reply_at=_optional_datetime(row, "last_reply_at", user_business_path),
        )
        for row in _read_csv(
            user_business_path,
            (
                "user_id",
                "business_id",
                "why_user_knows_account",
                "last_activity_at",
                "allows_promotions",
                "promotions_opted_out_at",
                "activity_count_180d",
                "messages_opened_30d",
                "messages_dismissed_30d",
                "messages_replied_30d",
                "last_reply_at",
            ),
        )
    ]

    loaded_history = _load_messages(history_path, HistoryMessage)
    message_history = [record for record in loaded_history if isinstance(record, HistoryMessage)]

    message_events = [
        MessageEvent(
            user_id=_required(row, "user_id", events_path),
            message_id=_required(row, "message_id", events_path),
            message_opened=_boolean(row, "message_opened", events_path),
            message_replied=_boolean(row, "message_replied", events_path),
            reaction_time_minutes=_optional_float(row, "reaction_time_minutes", events_path),
            notification_dismissed=_boolean(row, "notification_dismissed", events_path),
            muted_after_message=_boolean(row, "muted_after_message", events_path),
            message_reported=_boolean(row, "message_reported", events_path),
        )
        for row in _read_csv(
            events_path,
            (
                "user_id",
                "message_id",
                "message_opened",
                "message_replied",
                "reaction_time_minutes",
                "notification_dismissed",
                "muted_after_message",
                "message_reported",
            ),
        )
    ]

    images = [
        MediaRef(
            media_id=_required(row, "image_id", images_path),
            kind="image",
            path=(media_path := _media_path(_required(row, "file_path", images_path), root, images_path)),
            exists=True,
            sha256=_media_sha256(media_path),
        )
        for row in _read_csv(images_path, ("image_id", "file_path"))
    ]
    voice_notes = [
        MediaRef(
            media_id=_required(row, "voice_note_id", voice_notes_path),
            kind="voice",
            path=(media_path := _media_path(_required(row, "file_path", voice_notes_path), root, voice_notes_path)),
            exists=True,
            sha256=_media_sha256(media_path),
        )
        for row in _read_csv(voice_notes_path, ("voice_note_id", "file_path"))
    ]

    daily_notification_summary = [
        DailyNotificationSummary(
            user_id=_required(row, "user_id", daily_summary_path),
            date=_date(row, "date", daily_summary_path),
            notifications_sent=_integer(row, "notifications_sent", daily_summary_path),
            notifications_dismissed=_integer(row, "notifications_dismissed", daily_summary_path),
        )
        for row in _read_csv(
            daily_summary_path,
            ("user_id", "date", "notifications_sent", "notifications_dismissed"),
        )
    ]

    message_ids = [message.message_id for message in messages]
    if len(set(message_ids)) != len(message_ids):
        raise ValueError("messages.csv contains duplicate message IDs")
    if len(set(output_row_order)) != len(output_row_order):
        raise ValueError("output.csv contains duplicate message IDs")
    if set(output_row_order) != set(message_ids):
        raise ValueError("output.csv and messages.csv must contain the same message ID set")

    history_by_user_build: defaultdict[str, list[HistoryMessage]] = defaultdict(list)
    for history_message in message_history:
        history_by_user_build[history_message.user_id].append(history_message)
    history_by_user = {
        user_id: sorted(records, key=lambda record: (record.created_at, record.message_id))
        for user_id, records in history_by_user_build.items()
    }

    daily_summary_build: defaultdict[str, list[DailyNotificationSummary]] = defaultdict(list)
    for summary in daily_notification_summary:
        daily_summary_build[summary.user_id].append(summary)
    daily_summary_by_user = {
        user_id: sorted(records, key=lambda record: record.date)
        for user_id, records in daily_summary_build.items()
    }

    return Dataset(
        messages=messages,
        output_row_order=output_row_order,
        users=users,
        groups=groups,
        group_members=group_members,
        business_accounts=business_accounts,
        user_business_history=user_business_history,
        message_history=message_history,
        message_events=message_events,
        images=images,
        voice_notes=voice_notes,
        daily_notification_summary=daily_notification_summary,
        history_by_user=history_by_user,
        events_by_user_message=_unique_index(
            message_events,
            lambda event: (event.user_id, event.message_id),
            "message_events.csv",
        ),
        group_members_by_group_user=_unique_index(
            group_members,
            lambda member: (member.group_id, member.user_id),
            "group_members.csv",
        ),
        business_by_id=_unique_index(
            business_accounts,
            lambda business: business.business_id,
            "business_accounts.csv",
        ),
        user_business_by_user_business=_unique_index(
            user_business_history,
            lambda history: (history.user_id, history.business_id),
            "user_business_history.csv",
        ),
        images_by_id=_unique_index(images, lambda media: media.media_id, "images.csv"),
        voice_notes_by_id=_unique_index(voice_notes, lambda media: media.media_id, "voice_notes.csv"),
        daily_summary_by_user=daily_summary_by_user,
    )
