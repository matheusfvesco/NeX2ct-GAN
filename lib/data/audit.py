"""Versioned, deterministic JSON audit records and append-only event journal."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_SCHEMA_VERSION = 1
AUDIT_STATUSES = {
    "outside_all_splits",
    "patient_id_unresolved",
    "read_ok",
    "read_failed",
    "indexed",
    "eligible",
    "ineligible",
    "selected",
    "not_selected",
    "processed",
    "reused",
    "partially_processed",
    "processing_failed",
    "skipped",
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, value: dict | list) -> None:
    """Write stable UTF-8 JSON through a same-directory temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        default=_json_default,
    ) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def update_processing_results(path: Path, patient_results: dict, cohort) -> None:
    """Preserve prepare evidence while updating patient/file/series processing states."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    selected_paths = set()
    for row in payload["files"]:
        if row.get("selection_status") == "selected":
            selected_paths.add(row["path"])
            row["processing_status"] = _processing_status(row, patient_results)
    for row in payload["series"]:
        if set(row.get("file_paths", [])) & selected_paths:
            row["selection_status"] = "selected"
            row["processing_status"] = _processing_status(row, patient_results)
        elif row.get("eligibility_status") == "eligible":
            row.update(selection_status="not_selected", selection_reason="series_not_selected_by_manifest")
    for row in payload["patients"]:
        patient_id = row["patient_id"]
        row["processing_status"] = _processing_status(row, patient_results)
        row["destination_results"] = patient_results.get(patient_id, {})
        row["destination_status"] = {
            destination: patient_results.get(patient_id, {}).get(destination, {}).get("status", "not_started")
            for destination in cohort.destinations(patient_id)
        }
    write_json(path, payload)


def _processing_status(row, patient_results):
    destinations = patient_results.get(row.get("patient_id"), {})
    row["destination_results"] = destinations
    statuses = {result["status"] for result in destinations.values()}
    if not statuses:
        return "not_started"
    if statuses <= {"processed", "reused"}:
        return "reused" if statuses == {"reused"} else "processed"
    if statuses & {"processed", "reused"}:
        return "partially_processed"
    return "processing_failed"


class DicomAudit:
    """Collect one row per discovered file/series/patient plus processing events."""

    def __init__(self) -> None:
        self.files: list[dict] = []
        self.files_by_path: dict[str, dict] = {}
        self.series: list[dict] = []
        self.patients: list[dict] = []
        self.events: list[dict] = []

    def add_file(self, **record: Any) -> dict:
        row = {
            "read_status": "not_read",
            "eligibility_status": "not_evaluated",
            "selection_status": "not_evaluated",
            "processing_status": "not_started",
            "errors": [],
            **record,
        }
        for field in ("read_status", "eligibility_status", "selection_status", "processing_status"):
            status = row[field]
            if status in AUDIT_STATUSES:
                continue
            if status not in {"not_read", "not_evaluated", "not_started", "pending"}:
                raise ValueError(f"Unknown {field}: {status}")
        self.files.append(row)
        if row.get("path"):
            self.files_by_path[row["path"]] = row
        return row

    def add_series(self, **record: Any) -> dict:
        row = {"eligibility_status": "not_evaluated", "selection_status": "not_evaluated", "errors": [], **record}
        self.series.append(row)
        return row

    def add_patient(self, **record: Any) -> dict:
        row = {"destinations": [], "destination_status": {}, "errors": [], **record}
        self.patients.append(row)
        return row

    def add_event(self, **record: Any) -> dict:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attempt": 1,
            **record,
        }
        self.events.append(row)
        return row

    def to_dict(self) -> dict:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "files": sorted(self.files, key=lambda row: str(row.get("path", ""))),
            "series": sorted(self.series, key=lambda row: str(row.get("series_instance_uid") or row.get("path", ""))),
            "patients": sorted(self.patients, key=lambda row: str(row.get("patient_id", ""))),
        }

    def write(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    def write_events(self, path: str | Path) -> None:
        write_json(path, {"schema_version": AUDIT_SCHEMA_VERSION, "events": self.events})


class EventJournal:
    """Append-only JSONL journal for resumable, long-running patient processing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._attempts = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid event journal row {number}: {exc}") from exc
                    key = (record.get("patient_id"), record.get("destination"))
                    self._attempts[key] = max(self._attempts.get(key, 0), record.get("attempt", 0))

    def start_attempt(self, patient_id, destination, output_path):
        key = (patient_id, destination)
        attempt = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempt
        self.append({"patient_id": patient_id, "destination": destination, "stage": "generate",
                     "status": "started", "attempt": attempt, "output_path": str(output_path)})
        return attempt

    def append(self, event: dict) -> None:
        row = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attempt": 1,
            **event,
        }
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def consolidate(self, output_path: str | Path) -> list[dict]:
        records = []
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid event journal row {line_number}: {exc}") from exc
        records.sort(key=lambda row: (row.get("timestamp_utc", ""), row.get("patient_id", ""), row.get("destination", ""), row.get("attempt", 0)))
        write_json(output_path, {"schema_version": AUDIT_SCHEMA_VERSION, "events": records})
        return records
