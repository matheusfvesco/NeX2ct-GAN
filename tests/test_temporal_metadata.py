"""Synthetic metadata regressions; no image decoding or real dataset required."""

from datetime import date, datetime, timezone

import pytest

from lib.data.temporal import parse_acquisition_time
from lib.data.selection import select_exams, selected_intervals


@pytest.mark.parametrize(("raw", "precision", "parsed"), [
    ("2024", "year", None),
    ("202402", "month", None),
    ("20240229", "day", date(2024, 2, 29)),
    ("2024022912", "hour", datetime(2024, 2, 29, 12)),
    ("202402291234", "minute", datetime(2024, 2, 29, 12, 34)),
    ("20240229123456", "second", datetime(2024, 2, 29, 12, 34, 56)),
    ("20240229123456.1", "microsecond", datetime(2024, 2, 29, 12, 34, 56, 100000)),
])
def test_dt_precision_is_not_invented(raw, precision, parsed):
    result = parse_acquisition_time({"AcquisitionDateTime": raw})
    assert result["precision"] == precision
    assert result["parsed"] == parsed
    assert result["status"] != "invalid"


@pytest.mark.parametrize("raw", ["0000", "202413", "20230229", "2024010124", "202401011260", "202401011234.5", "20240101123456+1260", "20240101123456-1300", "20240101123456+1401"])
def test_invalid_partial_dates_times_and_offsets_are_reported(raw):
    result = parse_acquisition_time({"AcquisitionDateTime": raw})
    assert result["status"] == "invalid"
    assert result["parsed"] is None
    assert result["error"]


@pytest.mark.parametrize("source", [
    {"AcquisitionDateTime": "20240101123000+0200"},
    {"AcquisitionDateTime": "20240101123000", "TimezoneOffsetFromUTC": "+0200"},
    {"AcquisitionDate": "20240101", "AcquisitionTime": "123000", "TimezoneOffsetFromUTC": "+0200"},
])
def test_timezone_is_supported_for_each_acquisition_source(source):
    result = parse_acquisition_time(source)
    assert result["parsed"].astimezone(timezone.utc) == datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)


def test_source_conflict_is_visible_and_forces_temporal_fallback():
    record = parse_acquisition_time({"AcquisitionDateTime": "20240101120000", "AcquisitionDate": "20240102", "AcquisitionTime": "120000"})
    assert record["status"] == "conflict"
    assert len(record["sources"]) == 2
    ct = {"path": "ct", "slice_count": 30, "time_status": record["status"], "time_precision": record["precision"], "acquisition_datetime": record["parsed"]}
    view = {"path": "view", "acquisition_datetime": datetime(2024, 1, 1, 12), "time_precision": "second"}
    assert select_exams("temporal", [ct], [view], [view])["status"] == "temporal_fallback"


def test_consistent_sources_compare_at_common_precision_and_in_utc():
    record = parse_acquisition_time({"AcquisitionDateTime": "20240101120000.25+0200", "AcquisitionDate": "20240101", "AcquisitionTime": "100000", "TimezoneOffsetFromUTC": "+0000"})
    assert record["source_comparison"] == "consistent"


def test_bad_dt_falls_back_to_valid_da_tm_without_losing_error():
    record = parse_acquisition_time({"AcquisitionDateTime": "broken", "AcquisitionDate": "20240101", "AcquisitionTime": "12"})
    assert record["fallback_used"]
    assert record["precision"] == "hour"
    assert record["sources"][0]["status"] == "invalid"


def test_only_date_remains_date_and_intervals_use_days():
    record = parse_acquisition_time({"AcquisitionDate": "20240101"})
    assert type(record["parsed"]) is date
    selection = {"ct": {"acquisition_datetime": record["parsed"], "time_precision": "day"},
                 "frontal": {"acquisition_datetime": date(2024, 1, 3), "time_precision": "day"}}
    interval = selected_intervals(selection)["frontal_minus_ct"]
    assert (interval["value"], interval["unit"], interval["precision"]) == (2, "day", "day")


def test_missing_acquisition_never_uses_study_date_as_acquisition():
    record = parse_acquisition_time({"StudyDate": "20240101"})
    assert record["status"] == "missing"
    assert record["parsed"] is None
    assert record["alternatives"]["StudyDate"] == "20240101"
