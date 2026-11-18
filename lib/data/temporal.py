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
              "alternatives": {key: str(dataset.get(key)) for key in ("StudyDate", "StudyTime", "SeriesDate", "SeriesTime", "ContentDate", "ContentTime") if dataset.get(key)}}
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


OBSERVATION_FIELDS = (
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    "AcquisitionDateTime", "AcquisitionDate", "AcquisitionTime", "TimezoneOffsetFromUTC",
    "StudyDate", "StudyTime", "SeriesDate", "SeriesTime", "ContentDate", "ContentTime",
    "PatientPosition", "PatientOrientation", "TreatmentDate", "TreatmentTime",
    "ImagePositionPatient",
    "TreatmentMachineName", "RTPlanLabel", "RTPlanName",
    "NominalPercentageOfRespiratoryPhase", "StartingRespiratoryPhase", "EndingRespiratoryPhase",
)


def observe_dataset(dataset, path, *, ct_series=False):
    """Observe an already-read dataset; malformed optional tags cannot reject an image."""
    from lib.data.respiratory import AUXILIARY_TAGS, PHASE_TAGS, TEXT_FIELDS, extract_respiratory_evidence

    record = {"path": str(path), "metadata": {}, "field_status": {}, "metadata_errors": []}
    for field in OBSERVATION_FIELDS:
        if ct_series and field in {"SOPInstanceUID", "ImagePositionPatient"}:
            continue
        try:
            value = dataset.get(field)
            record["metadata"][field] = str(value) if value is not None else None
            record["field_status"][field] = "observed" if value is not None and str(value).strip() else "missing"
        except Exception as exc:
            record["metadata"][field] = None
            record["field_status"][field] = "invalid"
            record["metadata_errors"].append({"field": field, "message": str(exc)})
    for field, key in (("StudyInstanceUID", "study_instance_uid"),
                       ("SeriesInstanceUID", "series_instance_uid"),
                       ("SOPInstanceUID", "sop_instance_uid")):
        if field in record["metadata"]:
            record[key] = record["metadata"][field]
    record["temporal"] = parse_acquisition_time(record["metadata"])
    if ct_series:
        # Only general header fields: never traverse functional groups or frames.
        general_fields = {}
        for field in (*PHASE_TAGS, *AUXILIARY_TAGS, *TEXT_FIELDS):
            try:
                general_fields[field] = dataset.get(field)
            except Exception as exc:
                record["metadata_errors"].append({"field": field, "message": str(exc)})
        dataset = general_fields
    record["respiratory_evidence"] = extract_respiratory_evidence(dataset, path=str(path))
    return record


def observe_ct_series_header(dataset, series_path):
    """Represent one CT series from its already-loaded representative header."""
    record = observe_dataset(dataset, series_path, ct_series=True)
    record.update(role="ct_series", record_format="ct_series_header_v1",
                  header_source="existing_representative_header",
                  temporal_scope="series_assumed_from_representative_header",
                  series_temporal_policy="representative_header_applies_to_series")
    return record


def temporal_observation(candidate):
    """Restore a timestamp from raw DICOM fields after JSON serialization."""
    temporal = candidate.get("temporal") or {}
    raw_fields = temporal.get("raw_fields")
    if raw_fields:
        return parse_acquisition_time(raw_fields)
    metadata = candidate.get("metadata")
    if metadata:
        return parse_acquisition_time(metadata)
    value = candidate.get("acquisition_datetime")
    precision = candidate.get("time_precision")
    if isinstance(value, (date, datetime)):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = date.fromisoformat(value) if precision == "day" else datetime.fromisoformat(value)
        except ValueError:
            return {"parsed": None, "precision": precision, "source": candidate.get("time_source"),
                    "status": "invalid", "fallback_used": False, "raw_fields": {}}
    else:
        return parse_acquisition_time({})
    return {
        "parsed": parsed, "precision": precision or ("day" if type(parsed) is date else "second"),
        "source": candidate.get("time_source"), "status": candidate.get("time_status", "valid"),
        "fallback_used": False, "raw_fields": {},
        "resolution_seconds": candidate.get("time_resolution_seconds", 1),
    }


def observed_interval(ct, radiograph, evidence=None):
    """Calculate RX minus CT only with explicit, sourced cross-exam comparability."""
    result = {"status": "unavailable", "value": None, "unit": None, "precision": None}
    if not ct or not radiograph:
        return {**result, "reason": "selected_exam_missing"}
    if ct.get("record_format") not in {None, "ct_series_header_v1"}:
        return {**result, "reason": "unsupported_ct_record_format"}
    if ct.get("temporal_scope") not in {"series_assumed_from_representative_header", "series"}:
        return {**result, "reason": "ct_series_temporal_reference_not_defined"}
    left = temporal_observation(radiograph)
    right = temporal_observation(ct)
    result.update(left_source=left["source"], right_source=right["source"],
                  sign_convention="rx_minus_ct", comparability_evidence=evidence)
    if any(row["parsed"] is None or row["status"] in {"missing", "invalid", "conflict"}
           or row["fallback_used"] for row in (left, right)):
        return {**result, "reason": "timestamps_missing_invalid_or_conflicting"}
    if not evidence or not evidence.get("source") or evidence.get("calendar_comparable") is not True:
        return {**result, "reason": "calendar_comparability_not_demonstrated"}
    a, b = left["parsed"], right["parsed"]
    if left["precision"] != right["precision"]:
        return {**result, "reason": "incompatible_precision"}
    if left["precision"] == "day":
        return {**result, "status": "available", "reason": None, "kind": "calendar_date",
                "value": (a - b).days, "unit": "day", "precision": "day"}
    if evidence.get("instant_comparable") is not True or a.utcoffset() is None or b.utcoffset() is None:
        return {**result, "reason": "instant_comparability_not_demonstrated"}
    return {**result, "status": "available", "reason": None, "kind": "instant",
            "value": (a - b).total_seconds(), "unit": "second", "precision": left["precision"],
            "resolution_seconds": max(left["resolution_seconds"], right["resolution_seconds"])}
