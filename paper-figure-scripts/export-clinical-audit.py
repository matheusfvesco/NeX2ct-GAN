#!/usr/bin/env python3
"""Export one auditable clinical row per patient in the real split union."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data.clinical import validate_interventions
from lib.data.cohort import Cohort


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_hdf5_json(path: Path, key: str) -> dict | None:
    if not path.is_file():
        return None
    with h5py.File(path, "r") as source:
        if key not in source:
            return None
        value = source[key][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def _patient_hdf5_metadata(path):
    """A missing/corrupt product is a clinical row state, never a dropped patient."""
    if path is None:
        return None, None, {"status": "not_requested"}
    if not path.is_file():
        return None, None, {"status": "missing", "path": str(path)}
    try:
        metadata = _read_hdf5_json(path, "metadata_json")
        provenance = _read_hdf5_json(path, "provenance_json")
        status = "available" if metadata is not None and provenance is not None else "incomplete"
        return metadata, provenance, {"status": status, "path": str(path)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, None, {"status": "unreadable", "path": str(path),
                            "error": {"type": type(exc).__name__, "message": str(exc)}}


def export_clinical_audit(
    manifest_path: Path,
    audit_path: Path,
    events_path: Path,
    output_dir: Path,
    *,
    interventions_path: Path | None = None,
    hdf5_root: Path | None = None,
) -> list[dict]:
    manifest = _read_json(manifest_path, {})
    if manifest.get("schema_version") != 1 or "splits_dir" not in manifest:
        raise ValueError("A valid versioned selection manifest is required")
    cohort = Cohort(manifest["splits_dir"])
    if manifest.get("split_hashes") != {name: split.sha256 for name, split in cohort.splits.items()}:
        raise ValueError("Splits differ from the prepared cohort")
    audit = _read_json(audit_path, {})
    events = _read_json(events_path, {})
    interventions = _read_json(interventions_path, {}) if interventions_path else None
    audit_patients = {row.get("patient_id"): row for row in audit.get("patients", [])}
    event_by_patient = {}
    for event in events.get("events", []):
        event_by_patient.setdefault(event.get("patient_id"), []).append(event)
    real_ids = {
        item["patient_id"]
        for item in manifest.get("patients", [])
        if any(name.startswith("real_") for name in item.get("memberships", []))
    }
    if real_ids != set(cohort.real_ids):
        raise ValueError("Manifest real population does not match the authoritative splits")
    intervention_by_id = validate_interventions(interventions, real_ids)
    real_rows = [item for item in manifest.get("patients", []) if item.get("patient_id") in real_ids]
    if len({item["patient_id"] for item in real_rows}) != len(real_rows):
        raise ValueError("Selection manifest contains duplicate clinical patient IDs")

    rows = []
    for item in manifest.get("patients", []):
        memberships = item.get("memberships", [])
        if not any(name.startswith("real_") for name in memberships):
            continue
        selected = item.get("selected") or {}
        intervention = intervention_by_id[item["patient_id"]]
        selected_timestamps = [selected.get(role) for role in ("ct", "frontal", "lateral")]
        respiratory = item.get("respiratory_evidence", {})
        respiratory_statuses = {
            evidence.get("status")
            for role_data in respiratory.values()
            for evidence in (role_data.get("evidence", []) if isinstance(role_data, dict) else role_data)
        }
        real_destination = next((name.removesuffix(".txt") for name in memberships if name.startswith("real_")), None)
        h5_path = hdf5_root / "real" / real_destination / item["patient_id"] / "ct_xray_data.h5" if hdf5_root and real_destination else None
        metadata, provenance, hdf5_status = _patient_hdf5_metadata(h5_path)
        row = {
            "patient_id": item["patient_id"],
            "memberships": memberships,
            "selection_status": item.get("status"),
            "selection_policy": item.get("policy", item.get("requested_policy")),
            "fallback_reason": item.get("fallback_reason"),
            "ct": selected.get("ct"),
            "frontal": selected.get("frontal"),
            "lateral": selected.get("lateral"),
            "intervals": item.get("intervals", {}),
            "respiratory_evidence": item.get("respiratory_evidence", {}),
            "annotation_conflict": item.get("annotation_conflict", False),
            "audit_patient": audit_patients.get(item["patient_id"]),
            "processing_events": event_by_patient.get(item["patient_id"], []),
            "intervention": intervention,
            "hdf5_metadata": metadata,
            "hdf5_provenance": provenance,
            "hdf5_status": hdf5_status,
            "evaluability": {
                "ct_frontal_interval": bool(item.get("intervals", {}).get("frontal_minus_ct", {}).get("status") == "available"),
                "ct_lateral_interval": bool(item.get("intervals", {}).get("lateral_minus_ct", {}).get("status") == "available"),
                "frontal_lateral_interval": bool(item.get("intervals", {}).get("lateral_minus_frontal", {}).get("status") == "available"),
                "complete_three_image_pair": bool(selected.get("ct") and selected.get("frontal") and selected.get("lateral")),
                "all_selected_timestamps_available": bool(
                    all(candidate and candidate.get("acquisition_datetime") is not None
                        and candidate.get("time_status") in {"valid", "partial"} for candidate in selected_timestamps)
                ),
                "respiratory_evidence_available": bool(respiratory_statuses - {"unavailable", "unreadable", None}),
                "respiratory_phase_evaluable": bool(respiratory_statuses & {"explicit_dicom", "textual_indication"}),
                "intervention_context_assessed": intervention.get("status") != "not_assessed",
            },
        }
        rows.append(row)

    rows.sort(key=lambda row: row["patient_id"])
    cohort.verify_unchanged()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clinical_pairs.json").write_text(
        json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "patient_id", "memberships", "selection_status", "selection_policy", "fallback_reason",
        "ct", "frontal", "lateral", "intervals", "respiratory_evidence", "annotation_conflict", "audit_patient",
        "processing_events", "intervention", "hdf5_metadata", "hdf5_provenance", "hdf5_status", "evaluability",
    )
    with (output_dir / "clinical_pairs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})

    summary = {
        "schema_version": 1,
        "real_patient_count": len(rows),
        "complete_three_image_pair_count": sum(row["evaluability"]["complete_three_image_pair"] for row in rows),
        "ct_frontal_interval_evaluable_count": sum(row["evaluability"]["ct_frontal_interval"] for row in rows),
        "ct_lateral_interval_evaluable_count": sum(row["evaluability"]["ct_lateral_interval"] for row in rows),
        "frontal_lateral_interval_evaluable_count": sum(row["evaluability"]["frontal_lateral_interval"] for row in rows),
        "all_selected_timestamps_available_count": sum(row["evaluability"]["all_selected_timestamps_available"] for row in rows),
        "respiratory_evidence_available_count": sum(row["evaluability"]["respiratory_evidence_available"] for row in rows),
        "respiratory_phase_evaluable_count": sum(row["evaluability"]["respiratory_phase_evaluable"] for row in rows),
        "intervention_context_assessed_count": sum(row["evaluability"]["intervention_context_assessed"] for row in rows),
        "selection_status_counts": {},
        "intervention_status_counts": {},
    }
    for row in rows:
        status = str(row.get("selection_status") or "unknown")
        summary["selection_status_counts"][status] = summary["selection_status_counts"].get(status, 0) + 1
        status = str((row.get("intervention") or {}).get("status", "not_assessed"))
        summary["intervention_status_counts"][status] = summary["intervention_status_counts"].get(status, 0) + 1
    (output_dir / "clinical_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interventions", type=Path)
    parser.add_argument("--hdf5-root", type=Path)
    args = parser.parse_args(argv)
    export_clinical_audit(args.manifest, args.audit, args.events, args.output_dir, interventions_path=args.interventions, hdf5_root=args.hdf5_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
