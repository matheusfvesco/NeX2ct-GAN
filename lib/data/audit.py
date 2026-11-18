"""Versioned, deterministic JSON audit records and append-only event journal."""

from __future__ import annotations

import json
import os
import csv
from collections import defaultdict
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
    "excluded",
    "blocked",
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


def update_processing_results(path: str | Path, patient_results: dict, cohort=None) -> None:
    """Apply the DicomAudit update logic to a persisted audit document."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    audit = DicomAudit()
    audit.files, audit.series, audit.patients = payload["files"], payload["series"], payload["patients"]
    audit.record_processing_results(patient_results, cohort)
    write_json(path, payload)


def _processing_status(row, patient_results):
    destinations = patient_results.get(row.get("patient_id"), {})
    row["destination_results"] = destinations
    statuses = {result["status"] for result in destinations.values()}
    if not statuses or statuses == {"not_attempted"}:
        return "not_attempted"
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
        self.files_by_patient = defaultdict(list)
        self.series_by_patient = defaultdict(list)
        self.patients_by_id = {}
        self.series: list[dict] = []
        self.patients: list[dict] = []
        self.events: list[dict] = []
        self.errors: list[dict] = []
        self.samples: list[dict] = []

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
        self.files_by_patient[row.get("patient_id")].append(row)
        if row.get("path"):
            self.files_by_path[row["path"]] = row
        return row

    def add_series(self, **record: Any) -> dict:
        row = {"eligibility_status": "not_evaluated", "selection_status": "not_evaluated", "errors": [], **record}
        self.series.append(row)
        self.series_by_patient[row.get("patient_id")].append(row)
        return row

    def add_patient(self, **record: Any) -> dict:
        row = {"destinations": [], "destination_status": {}, "errors": [], **record}
        self.patients.append(row)
        self.patients_by_id[row.get("patient_id")] = row
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
            "audit_errors": self.errors,
            "samples": self.samples,
        }

    def select_ct_series(self, patient_id, path):
        """Record the existing CT choice before montage can fail, without opening files."""
        selected_path = str(Path(path).absolute())
        for row in self.series_by_patient[patient_id]:
            if row.get("role") != "ct_series" or row.get("eligibility_status") != "eligible":
                continue
            selected = row.get("path") == selected_path
            row.update(selection_status="selected" if selected else "not_selected",
                       processing_status="not_started" if selected else "not_applicable",
                       selection_reason=None if selected else "series_not_used_by_existing_processing_order")

    def record_processing_results(self, patient_results: dict, cohort=None) -> None:
        """Attach destination outcomes to the in-memory inventory records."""
        selected_paths = set()
        for row in self.files:
            if row.get("selection_status") == "selected":
                selected_paths.add(row["path"])
                row["processing_status"] = _processing_status(row, patient_results)
        for row in self.series:
            selected_ct = row.get("role") == "ct_series" and row.get("selection_status") == "selected"
            if selected_ct or set(row.get("file_paths", [])) & selected_paths:
                row.update(selection_status="selected", processing_status=_processing_status(row, patient_results))
            else:
                row["processing_status"] = "not_applicable"
                if row.get("eligibility_status") == "eligible":
                    row.update(selection_status="not_selected", selection_reason="series_not_selected_by_manifest")
        for row in self.patients:
            patient_id = row["patient_id"]
            results = patient_results.get(patient_id, {})
            row.update(
                processing_status=_processing_status(row, patient_results),
                destination_results=results,
                destination_status={
                    destination: results.get(destination, {}).get("status", "not_attempted")
                    for destination in (cohort.destinations(patient_id) if cohort else row.get("destinations", []))
                },
            )

    def persist_run_report(self, report: dict, audit_dir: str | Path) -> None:
        """Persist the audit tables and run-level coverage in one place."""
        audit_dir = Path(audit_dir)
        payload = self.to_dict()
        write_exam_rows(self, audit_dir / "exams.json", report.get("run_id"), payload=payload)
        self.write_csv(payload, audit_dir / "exams.csv", report.get("run_id"))
        now = datetime.now(timezone.utc).isoformat()
        report["updated_at_utc"] = now
        if report.get("status") != "running":
            report["finished_at_utc"] = now
        report.update(payload)
        report["audit_complete"] = not self.errors
        if self.errors and report.get("status") == "succeeded":
            report["status"] = "completed_with_audit_errors"
        self._record_coverage(report, payload)
        write_json(audit_dir / "run.json", report)

    @staticmethod
    def write_csv(payload, path, run_id=None):
        """Export the consolidated JSON snapshot without reading any source images."""
        series_by_file = {}
        for series in payload["series"]:
            for file_path in series.get("file_paths", []):
                series_by_file[file_path] = series
        exam_rows = []
        for file_row in payload["files"]:
            series = series_by_file.get(file_row.get("path"), {})
            exam_rows.append({**series, **file_row})
        exam_rows.extend(series for series in payload["series"] if not series.get("file_paths"))
        csv_rows = []
        for exam in exam_rows:
            row = {"run_id": run_id, **exam}
            source = row.get("source_metadata") or row.get("representative_metadata") or {}
            temporal = source.get("temporal") or row.get("temporal") or {}
            raw_fields = temporal.get("raw_fields", {})
            source_fields = source.get("metadata", {})
            field_status = source.get("field_status") or row.get("field_status", {})
            for field in (
                "AcquisitionDateTime", "AcquisitionDate", "AcquisitionTime",
                "TimezoneOffsetFromUTC", "StudyDate", "StudyTime", "SeriesDate",
                "SeriesTime", "ContentDate", "ContentTime", "PatientPosition",
                "PatientOrientation", "TreatmentDate", "TreatmentTime",
                "TreatmentMachineName", "RTPlanLabel", "RTPlanName",
                "NominalPercentageOfRespiratoryPhase", "StartingRespiratoryPhase",
                "EndingRespiratoryPhase", "ImagePositionPatient",
            ):
                row[field] = raw_fields.get(field, source_fields.get(field))
                row[f"{field}_status"] = field_status.get(field, "not_verified")
            if row.get("role") == "ct_series":
                row["SOPInstanceUID_status"] = "not_applicable"
                row["ImagePositionPatient_status"] = "not_applicable"
            row.update(
                StudyInstanceUID=source.get("study_instance_uid", row.get("study_instance_uid")),
                SeriesInstanceUID=source.get("series_instance_uid", row.get("series_instance_uid")),
                SOPInstanceUID=source.get("sop_instance_uid", row.get("sop_instance_uid")),
                temporal_source=temporal.get("source"),
                temporal_precision=temporal.get("precision"),
                temporal_status=temporal.get("status", "not_verified"),
            )
            if row.get("role") == "ct_series":
                for field, value in row.get("series_summary", {}).items():
                    row[field] = value
            csv_rows.append(row)
        required = {"run_id", "sample_id", "collection", "patient_id", "role", "path",
                    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
                    "AcquisitionDateTime", "AcquisitionDate", "AcquisitionTime",
                    "selection_status", "selection_reason", "processing_status", "metadata_status"}
        columns = sorted(required | {key for row in csv_rows for key in row if key != "file_paths"})
        with Path(path).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
                    if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                })
    @staticmethod
    def _record_coverage(report, payload):
        selection = report.get("selection", {})
        expected = {row["patient_id"] for row in selection.get("patients", [])}
        if not expected:
            subset = report.get("subset")
            if subset is not None:
                expected = set(subset.get("active_patient_ids", []))
            else:
                expected = {row["patient_id"] for row in report.get("cohort_snapshot", {}).get("memberships", [])}
        found = {row.get("patient_id") for row in payload["series"] if row.get("patient_id")}
        found.update(row.get("patient_id") for row in payload["files"] if row.get("patient_id"))
        extras = sorted({row.get("patient_id_raw") for row in payload["series"]
                         if row.get("reason") == "outside_all_splits" and row.get("patient_id_raw")})
        report["patient_coverage"] = {
            "expected_ids": sorted(expected), "found_expected_ids": sorted(expected & found),
            "missing_ids": sorted(expected - found) if report.get("inventory_complete") else [],
            "not_yet_verified_ids": sorted(expected - found) if not report.get("inventory_complete") else [],
            "unexpected_ids": extras,
        }
        metadata_coverage = {}
        metadata_fields = (
            "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
            "AcquisitionDateTime", "AcquisitionDate", "AcquisitionTime",
            "TimezoneOffsetFromUTC", "StudyDate", "StudyTime", "SeriesDate",
            "SeriesTime", "ContentDate", "ContentTime", "PatientPosition",
            "PatientOrientation", "TreatmentDate", "TreatmentTime",
            "TreatmentMachineName", "RTPlanLabel", "RTPlanName",
            "NominalPercentageOfRespiratoryPhase", "StartingRespiratoryPhase",
            "EndingRespiratoryPhase",
        )
        for row in payload["files"]:
            key = "|".join(str(row.get(field) or "unknown") for field in ("collection", "patient_id", "modality", "image"))
            group = metadata_coverage.setdefault(key, {
                "collection": row.get("collection"), "patient_id": row.get("patient_id"),
                "modality": row.get("modality"), "metadata_unit": "image",
                "records_in_normal_inventory": 0,
                "metadata_status_counts": {}, "field_status_counts": {},
            })
            group["records_in_normal_inventory"] += 1
            status = row.get("metadata_status", "not_verified")
            group["metadata_status_counts"][status] = group["metadata_status_counts"].get(status, 0) + 1
            source = row.get("source_metadata") or {}
            statuses = source.get("field_status", row.get("field_status", {}))
            for field in metadata_fields:
                field_status = statuses.get(field, "not_verified")
                counts = group["field_status_counts"].setdefault(field, {})
                counts[field_status] = counts.get(field_status, 0) + 1
        for row in payload["series"]:
            if row.get("role") != "ct_series":
                continue
            key = "|".join([*(str(row.get(field) or "unknown") for field in ("collection", "patient_id", "modality")), "ct_series"])
            group = metadata_coverage.setdefault(key, {
                "collection": row.get("collection"), "patient_id": row.get("patient_id"),
                "modality": row.get("modality"), "metadata_unit": "ct_series",
                "records_in_normal_inventory": 0,
                "metadata_status_counts": {}, "field_status_counts": {},
            })
            group["records_in_normal_inventory"] += 1
            status = row.get("metadata_status", "not_verified")
            group["metadata_status_counts"][status] = group["metadata_status_counts"].get(status, 0) + 1
            source = row.get("source_metadata") or row.get("representative_metadata") or {}
            for field in metadata_fields:
                if field == "SOPInstanceUID":
                    continue
                field_status = source.get("field_status", {}).get(field, "not_verified")
                counts = group["field_status_counts"].setdefault(field, {})
                counts[field_status] = counts.get(field_status, 0) + 1
        report["metadata_coverage"] = [metadata_coverage[key] for key in sorted(metadata_coverage)]
        report["metadata_coverage_scope"] = "observed_exam_units; CT series and RX images"
        report["ct_series_count"] = sum(row.get("role") == "ct_series" for row in payload["series"])

    def write(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    def write_events(self, path: str | Path) -> None:
        write_json(path, {"schema_version": AUDIT_SCHEMA_VERSION, "events": self.events})


def write_exam_rows(audit: DicomAudit, path: str | Path, run_id: str | None = None, *, payload: dict | None = None) -> None:
    """Write one filterable record per indexed/discovered exam series."""
    if payload is None:
        payload = audit.to_dict()
    files_by_path = {row.get("path"): row for row in payload["files"]}
    exams = []
    represented = set()
    for series in payload["series"]:
        row = dict(series)
        file_paths = row.get("file_paths", [])
        row["files"] = [files_by_path[file_path] for file_path in file_paths if file_path in files_by_path]
        represented.update(file_paths)
        read_statuses = {item.get("read_status") for item in row["files"]}
        row.setdefault("read_status", (
            "read_failed" if "read_failed" in read_statuses else
            "partially_read" if row.get("files_not_verified", 0) and "read_ok" in read_statuses else
            "read_ok" if read_statuses and read_statuses == {"read_ok"} else
            "indexed" if row["files"] else "not_read"
        ))
        if "expected_patient" not in row:
            row["expected_patient"] = bool(row.get("patient_id"))
        row["should_include"] = row.get("expected_patient") and row.get("eligibility_status") == "eligible"
        errors = row.get("errors", []) or [error for item in row["files"] for error in item.get("errors", [])]
        row["errors"] = errors
        row["error_type"] = errors[0].get("exception_type") if errors else None
        row["error_message"] = errors[0].get("message") if errors else row.get("reason")
        row["final_status"] = _exam_final_status(row)
        exams.append(row)
    for file_row in payload["files"]:
        if file_row.get("path") not in represented:
            row = {**file_row, "files": [file_row]}
            row["should_include"] = row.get("expected_patient") and row.get("eligibility_status") == "eligible"
            errors = row.get("errors", [])
            row["error_type"] = errors[0].get("exception_type") if errors else None
            row["error_message"] = errors[0].get("message") if errors else row.get("selection_reason")
            row["final_status"] = _exam_final_status(row)
            exams.append(row)
    for index, row in enumerate(exams):
        row.update(schema_version=AUDIT_SCHEMA_VERSION, run_id=run_id, record_id=index + 1)
        for field in ("path", "relative_path", "dataset", "patient_id", "patient_id_raw",
                      "study_instance_uid", "series_instance_uid", "error_type", "error_message"):
            row.setdefault(field, None)
    write_json(path, sorted(exams, key=lambda row: (str(row.get("dataset", "")),
                                                     str(row.get("path", "")),
                                                     str(row.get("series_instance_uid", "")))))


def _exam_final_status(row: dict) -> str:
    if (row.get("processing_status") == "processing_failed" or row.get("read_status") == "read_failed"
            or any(item.get("read_status") == "read_failed" for item in row.get("files", []))
            or row.get("eligibility_status") == "blocked" or row.get("errors")
            or (row.get("reason") or "").endswith("_failed")):
        return "failed"
    if row.get("selection_status") == "selected":
        status = row.get("processing_status")
        if status in {"processed", "reused"}:
            return "included"
        if status in {"not_attempted", "partially_processed"}:
            return status
        return "selected"
    if row.get("selection_status") == "not_selected":
        return "not_selected"
    if row.get("eligibility_status") == "ineligible" or row.get("selection_status") in {"skipped", "excluded"}:
        return "excluded"
    if row.get("reason") == "outside_all_splits":
        return "excluded"
    return "not_evaluated"

class EventJournal:
    """Track processing events in memory or in a persistent JSONL journal."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._attempts = {}
        self.events = []
        if self.path is not None and self.path.exists():
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
        self.events.append(row)
        if self.path is not None:
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def consolidate(self, output_path: str | Path) -> list[dict]:
        records = []
        if self.path is not None and self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid event journal row {line_number}: {exc}") from exc
        if self.path is None:
            records.extend(self.events)
        records.sort(key=lambda row: (row.get("timestamp_utc", ""), row.get("patient_id", ""), row.get("destination", ""), row.get("attempt", 0)))
        if output_path is not None:
            write_json(output_path, {"schema_version": AUDIT_SCHEMA_VERSION, "events": records})
        return records
