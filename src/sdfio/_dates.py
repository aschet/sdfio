# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The ``datetime`` field of the header: UTC in memory, the convention of the revision on disk."""

from __future__ import annotations

from datetime import UTC, datetime

from .revision import Revision

__all__ = ["format_sdf_datetime", "parse_sdf_datetime", "to_sdf_datetime"]


#: Fixed-width ``datetime`` field format, DDMMYYYYHHMM.
_DATETIME_FORMAT = "%d%m%Y%H%M"
# Neither revision defines a date that is unknown. Files in practice, BCR ones among them, use all
# zeros, which is no calendar date; it is read as "not recorded".
_UNSET_DATETIME = "0" * len("DDMMYYYYHHMM")


def to_sdf_datetime(value: datetime | None) -> datetime | None:
    """Return ``value`` as the timezone-aware UTC datetime that sdfio holds.

    ``None`` stays ``None``. The ``datetime`` field of the standard has the resolution of a
    minute, so the seconds and microseconds are dropped. A naive ``value`` is taken as local
    time, like :meth:`datetime.astimezone` does.

    >>> from datetime import timedelta, timezone
    >>> to_sdf_datetime(datetime(2024, 1, 1, 14, 0, 24, tzinfo=timezone(timedelta(hours=2))))
    datetime.datetime(2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    """
    if value is None:
        return None
    return value.astimezone(UTC).replace(second=0, microsecond=0)


def format_sdf_datetime(value: datetime | None, revision: Revision | None = None) -> str:
    """Format a datetime as the ``datetime`` field of the file.

    :param value: ``None`` becomes the all-zero "not recorded" placeholder. A naive value is
        local time. The seconds are dropped, as the field has the resolution of a minute.
    :param revision: A revision with :attr:`RevisionRules.requires_utc` gets the UTC clock.
        Another one, or none, gets the clock of the system timezone, see
        :func:`parse_sdf_datetime`.
    """
    utc = to_sdf_datetime(value)
    if utc is None:
        return _UNSET_DATETIME
    if revision is None or not revision.rules.requires_utc:
        utc = utc.astimezone()
    return utc.strftime(_DATETIME_FORMAT)


def parse_sdf_datetime(value: str, revision: Revision | None = None) -> datetime | None:
    """Parse the ``datetime`` field of a file; the result is UTC.

    :param revision: A revision with :attr:`RevisionRules.requires_utc` holds UTC. Another one,
        or none, holds the clock of the system timezone, which is converted. That is only right
        if the file is read in the timezone it was written in, because neither format records
        it. This is a limit of the format.
    :returns: ``None`` for the all-zero placeholder, and for a field that is no date of the form
        ``DDMMYYYYhhmm``, such as a blank one or a day 32. The date matters less than the
        heights, so such a file is read and the date is reported as not recorded.
    """
    stripped = value.strip()
    if (
        stripped == _UNSET_DATETIME
        or len(stripped) != len(_UNSET_DATETIME)
        or not stripped.isdigit()
    ):
        return None
    try:
        parsed = datetime.strptime(stripped, _DATETIME_FORMAT)
    except ValueError:  # a number that is no date, such as day 32 or month 13
        return None
    if revision is not None and revision.rules.requires_utc:
        return parsed.replace(tzinfo=UTC)
    return to_sdf_datetime(parsed)
