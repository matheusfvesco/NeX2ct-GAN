"""Tolerant acquisition timestamps with explicit precision and source conflicts.

Only acquisition attributes participate in selection. Calendar-only values are
dates, not acquisitions at midnight. Missing timezone remains unknown/local.
"""

import re
from datetime import date, datetime, timedelta, timezone


PRECISIONS = ("year", "month", "day", "hour", "minute", "second", "microsecond")
_DT = re.compile(r"(?P<calendar>\d{4}(?:\d{2}){0,5})(?:\.(?P<fraction>\d{1,6}))?(?P<offset>[+-]\d{4})?")
_TM = re.compile(r"\d{2}(?:\d{2}){0,2}(?:\.\d{1,6})?")


def _offset(text):
    if not text:
        return None
    if not re.fullmatch(r"[+-]\d{4}", text):
        raise ValueError(f"Invalid timezone offset {text!r}")
    hours, minutes = int(text[1:3]), int(text[3:5])
    total = hours * 60 + minutes
    if minutes > 59 or not (-720 <= (total if text[0] == "+" else -total) <= 840):
        raise ValueError(f"Timezone offset outside DICOM range: {text!r}")
    return timezone(timedelta(minutes=total if text[0] == "+" else -total))


def _parse_value(raw, inherited_offset):
    match = _DT.fullmatch(raw)
    if match is None:
        raise ValueError(f"Invalid DICOM acquisition value {raw!r}")
    calendar, fraction, explicit_offset = match.group("calendar", "fraction", "offset")
    if fraction and len(calendar) != 14:
        raise ValueError("Fractional time requires seconds")
    zone = _offset(explicit_offset or inherited_offset)
    parts = [int(calendar[:4])] + [int(calendar[index:index + 2]) for index in range(4, len(calendar), 2)]
    # Validate year/month as well as partial values without manufacturing a date.
    datetime(parts[0], parts[1] if len(parts) > 1 else 1, 1)
    precision = "microsecond" if fraction else PRECISIONS[len(parts) - 1]
    if len(parts) < 3:
        parsed = None
    elif len(parts) == 3:
        parsed = date(*parts)
    else:
        parsed = datetime(*(parts + [0] * (6 - len(parts))), microsecond=int((fraction or "0").ljust(6, "0")), tzinfo=zone)
    return {
        "raw": raw, "parsed": parsed, "precision": precision,
        "status": "partial" if len(parts) < 4 else "valid", "error": None,
        "timezone": (explicit_offset or inherited_offset) or None,
        "fractional_digits": len(fraction) if fraction else 0,
        "resolution_seconds": 10 ** -len(fraction) if fraction else {"hour": 3600, "minute": 60, "second": 1}.get(precision),
    }


def _read_source(raw, source, offset):
    try:
        record = _parse_value(raw, offset)
    except (ValueError, OverflowError, TypeError) as exc:
        record = {"raw": raw, "parsed": None, "precision": None, "status": "invalid",
                  "error": f"{type(exc).__name__}: {exc}", "timezone": None}
    return {"source": source, **record}


def _source_comparison(left, right):
    """Compare at their common precision, declining unknown timezone alignment."""
    a, b = left["parsed"], right["parsed"]
    precision = min(PRECISIONS.index(left["precision"]), PRECISIONS.index(right["precision"]))
    if precision < 2:
        width = 4 if precision == 0 else 6
        return "consistent" if left["raw"][:width] == right["raw"][:width] else "conflict"
    if a is None or b is None:
        return "incomparable"
    if precision <= 2:
        return "consistent" if (a.year, a.month, a.day) == (b.year, b.month, b.day) else "conflict"
    if (a.utcoffset() is None) != (b.utcoffset() is None):
        return "incomparable_timezone"
    if a.utcoffset() is not None:
        a, b = a.astimezone(timezone.utc), b.astimezone(timezone.utc)
    if precision == 6:
        digits = min(left["fractional_digits"], right["fractional_digits"])
        scale = 10 ** (6 - digits)
        a = a.replace(microsecond=a.microsecond // scale * scale)
        b = b.replace(microsecond=b.microsecond // scale * scale)
    components = lambda value: (value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond)
    count = precision + 1
    return "consistent" if components(a)[:count] == components(b)[:count] else "conflict"


def parse_acquisition_time(dataset):
    """Prefer DT, fall back to DA/TM if invalid, expose disagreement and raw sources."""
    dt = str(dataset.get("AcquisitionDateTime") or "").strip()
    da = str(dataset.get("AcquisitionDate") or "").strip()
    tm = str(dataset.get("AcquisitionTime") or "").strip()
    offset = str(dataset.get("TimezoneOffsetFromUTC") or "").strip()
    sources = []
    if dt:
        sources.append(_read_source(dt, "AcquisitionDateTime", offset))
    if da or tm:
        raw = da + tm
        if (tm and (len(da) != 8 or not da.isdigit() or _TM.fullmatch(tm) is None)) or len(da) not in {4, 6, 8}:
            sources.append({"source": "AcquisitionDate+AcquisitionTime", "raw": raw,
                            "parsed": None, "precision": None, "status": "invalid",
                            "error": "Invalid AcquisitionDate/AcquisitionTime combination", "timezone": None})
        else:
            sources.append(_read_source(raw, "AcquisitionDate+AcquisitionTime" if tm else "AcquisitionDate", offset))
    usable = [source for source in sources if source["status"] != "invalid"]
    resolved = [source for source in usable if source["parsed"] is not None]
    chosen = (resolved or usable)[0] if usable else (sources[0] if sources else {
        "raw": None, "parsed": None, "precision": None, "source": None,
        "status": "missing", "error": None, "timezone": None,
    })
    try:
        comparison = _source_comparison(*usable[:2]) if len(usable) > 1 else "single_source"
    except (ValueError, OverflowError):
        comparison = "incomparable_timezone"
    result = {**chosen, "sources": sources, "source_comparison": comparison,
              "fallback_used": bool(sources and chosen is not sources[0]),
              "raw_fields": {"AcquisitionDateTime": dt or None, "AcquisitionDate": da or None,
                             "AcquisitionTime": tm or None, "TimezoneOffsetFromUTC": offset or None},
              "alternatives": {key: str(dataset.get(key)) for key in ("StudyDate", "StudyTime", "SeriesDate", "SeriesTime") if dataset.get(key)}}
    if comparison == "conflict":
        result.update(status="conflict", error="Acquisition sources disagree at their common precision")
    return result


def comparable_datetime(candidate):
    """Internal arithmetic value; precision and status remain mandatory companions."""
    if candidate.get("time_status") in {"invalid", "conflict", "missing"}:
        return None
    value = candidate.get("acquisition_datetime")
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return None


def acquisition_distance(left, right):
    """Safe ordering distance for legacy directory callers with optional times."""
    a = comparable_datetime({"acquisition_datetime": left.get("parsed"), "time_status": left.get("status")})
    b = comparable_datetime({"acquisition_datetime": right.get("parsed"), "time_status": right.get("status")})
    precisions = {left.get("precision"), right.get("precision")}
    compatible = len(precisions) == 1 or precisions <= {"second", "microsecond"}
    if precisions == {"day"} and left.get("timezone") != right.get("timezone"):
        compatible = False
    if a is None or b is None or not compatible:
        return float("inf")
    if (a.utcoffset() is None) != (b.utcoffset() is None):
        return float("inf")
    return abs((a - b).total_seconds())
