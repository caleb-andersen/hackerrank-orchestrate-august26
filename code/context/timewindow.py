"""Do-not-disturb window parsing and evaluation."""

from datetime import datetime, time


def parse_dnd_window(raw: str | None) -> tuple[time, time] | None:
    """Parse an HH:MM-HH:MM window, returning None for invalid input."""
    if raw is None:
        return None
    parts = raw.strip().split("-")
    if len(parts) != 2:
        return None
    try:
        return (
            datetime.strptime(parts[0].strip(), "%H:%M").time(),
            datetime.strptime(parts[1].strip(), "%H:%M").time(),
        )
    except ValueError:
        return None


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def dnd_state(
    created_at: datetime,
    window: tuple[time, time] | None,
) -> tuple[bool, int | None]:
    """Evaluate a half-open DND interval in naive local wall-clock time."""
    if window is None:
        return (False, None)
    start, end = window
    local_time = created_at.time()
    if start == end:
        in_dnd = False
    elif start < end:
        in_dnd = start <= local_time < end
    else:
        in_dnd = local_time >= start or local_time < end
    if not in_dnd:
        return (False, None)
    minutes = (_minutes(end) - _minutes(local_time)) % (24 * 60)
    return (True, minutes)
