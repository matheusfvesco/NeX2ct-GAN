"""Deterministic, metadata-only CT/radiograph selection policies."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable
from lib.data.temporal import acquisition_distance, comparable_datetime


POLICIES = {"temporal", "largest_valid"}


def _stable(candidate: dict) -> tuple[str, str, str]:
    return (
        str(candidate.get("series_instance_uid") or ""),
        str(candidate.get("sop_instance_uid") or ""),
        str(candidate.get("path") or ""),
    )


def _time(candidate: dict) -> datetime | None:
    return comparable_datetime(candidate)


def _same_precision(candidates: Iterable[dict]) -> bool:
    values = list(candidates)
    times = [_time(value) for value in values]
    if any(value is None for value in times):
        return False
    precisions = {value.get("time_precision", "second") for value in values}
    zones = {(_time(value).utcoffset() is not None) for value in values}
    fractions = {value.get("time_fractional_digits", 0) for value in values}
    if precisions == {"day"} and len({value.get("time_timezone") for value in values}) != 1:
        return False
    return len(precisions) == 1 and len(zones) == 1 and len(fractions) == 1 and None not in precisions


def _distance_seconds(left: dict, right: dict) -> float:
    return abs((_time(left) - _time(right)).total_seconds())


def selected_intervals(selection: dict) -> dict:
    """Report signed intervals for any policy, including explicit missing reasons."""
    intervals = {}
    for name, left_role, right_role in (
        ("frontal_minus_ct", "frontal", "ct"),
        ("lateral_minus_ct", "lateral", "ct"),
        ("lateral_minus_frontal", "lateral", "frontal"),
    ):
        left, right = selection.get(left_role), selection.get(right_role)
        row = {"value": None, "unit": None, "precision": None, "status": "unavailable"}
        if left is None or right is None:
            row["reason"] = "selected_exam_missing"
        elif not _same_precision((left, right)):
            row["reason"] = "timestamps_missing_or_incomparable"
        else:
            precision = left.get("time_precision", "second")
            seconds = (_time(left) - _time(right)).total_seconds()
            row.update(value=seconds / 86400 if precision == "day" else seconds,
                       unit="day" if precision == "day" else "second",
                       precision=precision, status="available", reason=None)
            if precision == "microsecond":
                digits = left.get("time_fractional_digits", 0)
                row.update(fractional_digits=digits or None, resolution_seconds=10 ** -digits if digits else None)
        row.update(left_source=left.get("time_source") if left else None,
                   right_source=right.get("time_source") if right else None,
                   sign_convention=f"{left_role}_minus_{right_role}")
        intervals[name] = row
    return intervals


def select_largest_valid(cts: Iterable[dict], frontals: Iterable[dict], laterals: Iterable[dict]) -> dict:
    """Preserve legacy directory-time ordering and complete-multiview preference."""
    frontals, laterals = list(frontals), list(laterals)
    valid_cts = sorted(
        (item for item in cts if item.get("eligible", True)),
        key=_largest_ct_rank,
    )
    if not valid_cts:
        return {"policy": "largest_valid", "status": "no_valid_ct", "ct": None, "frontal": None, "lateral": None}
    ct = valid_cts[0]

    selected = {"frontal": None, "lateral": None}
    groups = {}
    for role, candidates in (("frontal", frontals), ("lateral", laterals)):
        singles = []
        for candidate in candidates:
            if not candidate.get("eligible", True):
                continue
            if candidate.get("source_type") == "multiview":
                group = groups.setdefault(candidate["group_path"], {"frontal": [], "lateral": []})
                group[role].append(candidate)
            else:
                singles.append(candidate)
        if singles:
            selected[role] = min(singles, key=lambda item: _legacy_view_rank(ct, item))
    if groups:
        complete = [name for name, group in groups.items() if group["frontal"] and group["lateral"]]
        pool = complete or list(groups)
        group_name = min(pool, key=lambda name: (
            _legacy_view_rank(ct, min(groups[name]["frontal"] + groups[name]["lateral"], key=_legacy_instance_rank)), name,
        ))
        for role in selected:
            if groups[group_name][role]:
                selected[role] = min(groups[group_name][role], key=_legacy_instance_rank)

    return {
        "policy": "largest_valid",
        "status": "selected",
        "fallback_reason": None,
        "ct": ct,
        **selected,
        "tie_break": "slice_count_desc_then_valid_time_uid_path; complete_multiview_then_directory_time; instance_path_uid",
        "view_ranking_reference": "source_directory_representative_acquisition",
    }


def _largest_ct_rank(candidate):
    timestamp = _time(candidate)
    ordinal = float("inf")
    if timestamp is not None:
        ordinal = (timestamp.replace(tzinfo=None) - datetime(1, 1, 1)).total_seconds()
        if timestamp.utcoffset() is not None:
            ordinal -= timestamp.utcoffset().total_seconds()
    return (-int(candidate.get("slice_count", 0)), timestamp is None,
            ordinal, _stable(candidate))


def _legacy_instance_rank(candidate):
    return (str(candidate.get("path", "")), _stable(candidate))


def _legacy_view_rank(ct, candidate):
    reference = {
        "acquisition_datetime": candidate.get("group_acquisition_datetime", candidate.get("acquisition_datetime")),
        "time_precision": candidate.get("group_time_precision", candidate.get("time_precision", "second")),
        "time_status": candidate.get("group_time_status", candidate.get("time_status")),
        "time_fractional_digits": candidate.get("group_time_fractional_digits", candidate.get("time_fractional_digits", 0)),
        "time_timezone": candidate.get("group_time_timezone", candidate.get("time_timezone")),
    }
    distance = acquisition_distance(
        {"parsed": ct.get("acquisition_datetime"), "precision": ct.get("time_precision", "second"), "status": ct.get("time_status"), "timezone": ct.get("time_timezone")},
        {"parsed": reference["acquisition_datetime"], "precision": reference["time_precision"], "status": reference["time_status"], "timezone": reference["time_timezone"]},
    )
    return (distance, str(candidate.get("group_path", candidate.get("path", ""))), _legacy_instance_rank(candidate))


def select_temporal(cts: Iterable[dict], frontals: Iterable[dict], laterals: Iterable[dict]) -> dict:
    """Minimize the documented temporal tuple; fallback explicitly when timestamps incomparable."""
    valid_cts = [item for item in cts if item.get("eligible", True)]
    valid_frontals = [item for item in frontals if item.get("eligible", True)]
    valid_laterals = [item for item in laterals if item.get("eligible", True)]
    if not valid_cts or not valid_frontals or not valid_laterals:
        return {
            **select_largest_valid(valid_cts, valid_frontals, valid_laterals),
            "requested_policy": "temporal",
            "status": "temporal_fallback",
            "fallback_reason": "complete_temporal_triplet_unavailable",
        }

    best = None
    for ct in sorted(valid_cts, key=_stable):
        comparable_frontals = [item for item in valid_frontals if _same_precision((ct, item))]
        comparable_laterals = [item for item in valid_laterals if _same_precision((ct, item))]
        if not comparable_frontals or not comparable_laterals:
            continue
        closest_front_distance = min(_distance_seconds(item, ct) for item in comparable_frontals)
        closest_lateral_distance = min(_distance_seconds(item, ct) for item in comparable_laterals)
        closest_frontals = [item for item in comparable_frontals if _distance_seconds(item, ct) == closest_front_distance]
        closest_laterals = [item for item in comparable_laterals if _distance_seconds(item, ct) == closest_lateral_distance]
        # At most two distinct times can be equidistant from a CT. At the same
        # time only the stable minimum can win; avoid quadratic duplicate ties.
        closest_frontals = _distinct_time_minima(closest_frontals)
        closest_laterals = _distinct_time_minima(closest_laterals)
        # Independent minima minimize the first two score components; only exact ties
        # need pairing to minimize frontal-lateral distance.
        for frontal in sorted(closest_frontals, key=_stable):
            for lateral in sorted(closest_laterals, key=_stable):
                dt_f = closest_front_distance
                dt_l = closest_lateral_distance
                dt_fl = _distance_seconds(lateral, frontal)
                score = (
                    max(dt_f, dt_l),
                    dt_f + dt_l,
                    dt_fl,
                    -int(ct.get("slice_count", 0)),
                    _stable(ct),
                    _stable(frontal),
                    _stable(lateral),
                )
                if best is None or score < best[0]:
                    best = (score, ct, frontal, lateral)

    if best is None:
        return {
            **select_largest_valid(valid_cts, valid_frontals, valid_laterals),
            "requested_policy": "temporal",
            "status": "temporal_fallback",
            "fallback_reason": "timestamps_missing_invalid_or_incomparable_precision",
        }

    _, ct, frontal, lateral = best
    result = {
        "policy": "temporal",
        "status": "selected",
        "fallback_reason": None,
        "ct": ct,
        "frontal": frontal,
        "lateral": lateral,
        "tie_break": "max_delta_then_sum_deltas_then_frontal_lateral_delta_then_slice_count_desc_then_stable_keys",
    }
    result["intervals"] = selected_intervals(result)
    return result


def select_exams(policy: str, cts: Iterable[dict], frontals: Iterable[dict], laterals: Iterable[dict]) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {sorted(POLICIES)}, got {policy!r}")
    if policy == "temporal":
        return select_temporal(cts, frontals, laterals)
    return select_largest_valid(cts, frontals, laterals)


def _distinct_time_minima(candidates):
    by_time = {}
    for candidate in candidates:
        timestamp = _time(candidate)
        current = by_time.get(timestamp)
        if current is None or _stable(candidate) < _stable(current):
            by_time[timestamp] = candidate
    return list(by_time.values())
