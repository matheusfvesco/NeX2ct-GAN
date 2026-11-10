from datetime import datetime, timedelta
from itertools import product
from random import Random

import pytest

from lib.data.selection import select_exams


BASE = datetime(2024, 1, 1, 12)


def candidate(path, minutes, **fields):
    return {"path": path, "acquisition_datetime": BASE + timedelta(minutes=minutes),
            "time_precision": "second", "time_status": "valid", "eligible": True, **fields}


def legacy_multiview(name, reference_minutes, frontal_minutes, lateral_minutes):
    group = {"group_path": name, "source_type": "multiview",
             "group_acquisition_datetime": BASE + timedelta(minutes=reference_minutes),
             "group_time_precision": "second", "group_time_status": "valid"}
    return (candidate(f"{name}/f.dcm", frontal_minutes, **group),
            candidate(f"{name}/l.dcm", lateral_minutes, **group))


def test_largest_valid_orders_multiview_by_directory_reference_not_pair_score():
    ct = candidate("ct", 0, slice_count=30)
    old_preferred = legacy_multiview("a", 1, -50, 1)
    different_pair = legacy_multiview("b", 2, -2, 2)
    result = select_exams("largest_valid", [ct], [old_preferred[0], different_pair[0]], [old_preferred[1], different_pair[1]])
    assert result["frontal"]["path"] == "a/f.dcm"
    assert result["lateral"]["path"] == "a/l.dcm"


def test_largest_valid_keeps_complete_multiview_priority_over_closer_singles():
    frontal, lateral = legacy_multiview("multiview", 100, 100, 101)
    result = select_exams("largest_valid", [candidate("ct", 0, slice_count=30)],
                          [candidate("single-f", 0), frontal], [candidate("single-l", 0), lateral])
    assert result["frontal"] is frontal
    assert result["lateral"] is lateral


def test_incomplete_multiview_overrides_only_its_available_role_like_legacy():
    frontal, _ = legacy_multiview("multiview", 10, 10, 11)
    single_lateral = candidate("single-l", 0)
    result = select_exams("largest_valid", [candidate("ct", 0, slice_count=30)],
                          [candidate("single-f", 0), frontal], [single_lateral])
    assert result["frontal"] is frontal
    assert result["lateral"] is single_lateral


def test_largest_ct_ties_are_stable_and_do_not_change_largest_scan_criterion():
    ct_a = candidate("ct-a", 0, slice_count=30)
    ct_b = candidate("ct-b", 0, slice_count=31)
    assert select_exams("largest_valid", [ct_a, ct_b], [], [])["ct"] is ct_b
    ct_b["slice_count"] = 30
    assert select_exams("largest_valid", [ct_b, ct_a], [], [])["ct"] is ct_a


@pytest.mark.parametrize("seed", range(12))
def test_temporal_matches_independent_exhaustive_oracle(seed):
    rng = Random(seed)
    cts = [candidate(f"ct-{i}", rng.randint(-10, 10), slice_count=30 + i) for i in range(4)]
    frontals = [candidate(f"f-{i}", rng.randint(-10, 10)) for i in range(6)]
    laterals = [candidate(f"l-{i}", rng.randint(-10, 10)) for i in range(6)]
    def score(triplet):
        ct, frontal, lateral = triplet
        f = abs((frontal["acquisition_datetime"] - ct["acquisition_datetime"]).total_seconds())
        l = abs((lateral["acquisition_datetime"] - ct["acquisition_datetime"]).total_seconds())
        fl = abs((lateral["acquisition_datetime"] - frontal["acquisition_datetime"]).total_seconds())
        return max(f, l), f + l, fl, -ct["slice_count"], ct["path"], frontal["path"], lateral["path"]
    expected = min(product(cts, frontals, laterals), key=score)
    rng.shuffle(cts)
    rng.shuffle(frontals)
    rng.shuffle(laterals)
    actual = select_exams("temporal", cts, frontals, laterals)
    assert tuple(actual[role]["path"] for role in ("ct", "frontal", "lateral")) == tuple(item["path"] for item in expected)


def test_unknown_timezone_and_known_timezone_do_not_compare():
    ct = candidate("ct", 0, slice_count=30)
    from datetime import timezone
    view = candidate("view", 0)
    view["acquisition_datetime"] = view["acquisition_datetime"].replace(tzinfo=timezone.utc)
    assert select_exams("temporal", [ct], [view], [view])["status"] == "temporal_fallback"
