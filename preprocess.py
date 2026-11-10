"""Prepare split-authorized DICOM selections and generate their HDF5 products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lib.data.audit import DicomAudit, EventJournal, update_processing_results, write_json
from lib.data.cohort import Cohort
from lib.data.dicom import DicomDataset
from lib.data.selection import select_exams, selected_intervals
from lib.data.respiratory import audit_selected_exams
from lib.data.validation import validate_split_outputs
from lib.data.generation import PROCESSING_CONTRACT_VERSION, destination_root, process_patient
from lib.dicom import DicomFile
from lib.logging import setup_logger
from lib.utils.dicom import XRayJsonClassifier


def _acquisition_fields(value) -> dict:
    temporal = getattr(value, "acquisition_time", {}) or {}
    return {
        "acquisition_datetime": temporal.get("parsed"),
        "time_precision": temporal.get("precision"),
        "time_source": temporal.get("source"),
        "time_status": temporal.get("status"),
        "time_fractional_digits": temporal.get("fractional_digits", 0),
        "time_timezone": temporal.get("timezone"),
        "acquisition_raw": temporal.get("raw"),
        "acquisition_time": temporal,
    }


def _candidate(value, role: str, audit: DicomAudit, *, group_path: str | None = None,
               source_type: str = "singleview", group_time: dict | None = None) -> dict:
    path = getattr(value, "path_directory", None)
    if path is None:
        path = getattr(value, "dicom_path", None)
    path = Path(path)
    paths = [Path(item) for item in getattr(value, "_paths_to_dicom_files", [path])]
    audit_by_path = audit.files_by_path
    instances = []
    for instance_path in sorted(paths):
        row = audit_by_path.get(str(instance_path.resolve()), {})
        instances.append(
            {
                "path": str(instance_path),
                "sop_instance_uid": row.get("sop_instance_uid"),
                "series_instance_uid": row.get("series_instance_uid"),
                "study_instance_uid": row.get("study_instance_uid"),
            }
        )
    if role != "ct":
        group_time = group_time or {}
    else:
        group_time = getattr(value, "acquisition_time", {}) or {}
    if hasattr(value, "is_valid_axial_scan") and not value.is_valid_axial_scan:
        eligible = False
    elif hasattr(value, "is_valid") and not value.is_valid:
        eligible = False
    else:
        eligible = True
    return {
        "role": role,
        "group_path": group_path or str(path),
        "source_type": source_type,
        "group_acquisition_datetime": group_time.get("parsed"),
        "group_time_precision": group_time.get("precision"),
        "group_time_status": group_time.get("status"),
        "group_time_fractional_digits": group_time.get("fractional_digits", 0),
        "group_time_timezone": group_time.get("timezone"),
        "view_classification_source": getattr(value, "view_type_source", None),
        "path": str(path),
        "paths": [item["path"] for item in instances],
        "instances": instances,
        "series_instance_uid": next((item["series_instance_uid"] for item in instances if item["series_instance_uid"]), None),
        "sop_instance_uid": next((item["sop_instance_uid"] for item in instances if item["sop_instance_uid"]), None),
        "slice_count": len(paths) if role == "ct" else 1,
        "eligible": eligible,
        **_acquisition_fields(value),
        "_object": value,
    }


def _patient_candidates(patient_data: dict, audit: DicomAudit) -> tuple[list, list, list]:
    cts = [_candidate(value, "ct", audit) for value in patient_data["cts"]]
    views = {"frontal": [], "lateral": []}
    rows = audit.files_by_path
    for directory in (*patient_data["xrays"], *patient_data["multiview"]):
        source = "multiview" if directory.is_multiview else "singleview"
        for path in directory._paths_to_dicom_files:
            try:
                view_type = XRayJsonClassifier().classify(path)
            except (FileNotFoundError, KeyError, ValueError):
                view_type = None
            row = rows.get(str(Path(path).resolve()))
            try:
                view = DicomFile(path)
                view_type = view_type or view.view_type
                if view.is_valid:
                    if view_type in views:
                        views[view_type].append(_candidate(
                        view, view_type, audit,
                        group_path=str(directory.path_directory), source_type=source,
                        group_time=directory.acquisition_time,
                    ))
                    elif row is not None:
                        row.update(eligibility_status="ineligible", selection_status="skipped",
                                   eligibility_reason="view_classification_unresolved",
                                   selection_reason="view_classification_unresolved")
                elif row is not None:
                    reason = "invalid_radiograph_type" if not view.is_valid else "view_classification_unresolved"
                    row.update(eligibility_status="ineligible", eligibility_reason=reason,
                               selection_status="skipped", selection_reason=reason)
            except Exception as exc:
                if row is not None:
                    row.update(eligibility_status="ineligible", selection_status="skipped",
                               eligibility_reason="view_classification_failed", selection_reason="view_classification_failed")
                    row["errors"].append({"stage": "view_classification", "code": "view_classification_failed",
                                          "exception_type": type(exc).__name__, "message": str(exc)})
    return cts, views["frontal"], views["lateral"]


def _manifest_candidate(candidate: dict | None) -> dict | None:
    if candidate is None:
        return None
    result = {key: value for key, value in candidate.items() if key != "_object"}
    return json.loads(json.dumps(result, default=lambda item: item.isoformat()))


def _selection_fingerprint(entry: dict) -> str:
    payload = {
        "policy": entry.get("policy"),
        "selected": entry.get("selected", {}),
        "intervals": entry.get("intervals", {}),
        "memberships": entry.get("memberships", []),
        "respiratory_evidence": entry.get("respiratory_evidence", {}),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_rows(items: list[dict]) -> list[dict]:
    return [_manifest_candidate(item) for item in sorted(items, key=lambda row: (row.get("path", ""), row.get("role", "")))]


def _discover_datasets(data_root: Path, dataset_names: list[str] | None) -> list[str]:
    if dataset_names:
        return sorted(set(dataset_names))
    return sorted(path.name for path in data_root.iterdir() if path.is_dir())


def prepare(args) -> dict:
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    cohort = Cohort(Path(args.splits_dir).resolve())
    run_dir = output_root
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cohort.write_snapshot(audit_dir / "cohort_snapshot.json")

    audit = DicomAudit()
    logger = setup_logger("preprocess-prepare", experiment_name="prepare", filename_comment="prepare")
    datasets = _discover_datasets(data_root, args.dataset)
    if not datasets:
        raise ValueError(f"No dataset directories found under {data_root}")
    annotation_path = data_root / "patient-selection-annotation.json"
    with annotation_path.open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    XRayJsonClassifier(root_data=data_root)
    dataset = DicomDataset(data_root, accepted_datasets=datasets, logger=logger, cohort=cohort, audit=audit)
    dataset.patient_selection_annotations = annotations
    dataset.load()
    # Preserve the existing inexpensive per-patient availability reports.
    if dataset.dataset:
        dataset.write_summary(output_root / "raw_data_summary.csv")
        dataset.write_counts(output_root / "modality_counts.csv")

    files_by_patient = {}
    for row in audit.files:
        files_by_patient.setdefault(row.get("patient_id"), []).append(row)
    patients = []
    for patient_id in cohort.all_ids:
        membership = list(cohort.memberships[patient_id])
        entry = {
            "patient_id": patient_id,
            "memberships": membership,
            "destinations": list(cohort.destinations(patient_id)),
            "status": "no_valid_candidates",
            "selection_by_destination": {},
            "annotation": annotations.get(patient_id),
            "annotation_conflict": patient_id in annotations and annotations[patient_id] != "selected",
        }
        if patient_id in dataset.dataset:
            cts, frontals, laterals = _patient_candidates(dataset[patient_id], audit)
            is_real = any(name.startswith("real_") for name in membership)
            requested_policy = (args.ct_selection or "temporal") if is_real else "largest_valid"
            selection = select_exams(
                requested_policy,
                cts,
                frontals if is_real else [],
                laterals if is_real else [],
            )
            if is_real:
                selection["intervals"] = selected_intervals(selection)
            selected_ct = selection.get("ct")
            selected_frontal = selection.get("frontal")
            selected_lateral = selection.get("lateral")
            selected_paths = {
                instance.get("path")
                for candidate in (selected_ct, selected_frontal, selected_lateral)
                if candidate
                for instance in candidate.get("instances", [])
            }
            candidate_paths = {
                instance.get("path")
                for candidate in (*cts, *frontals, *laterals)
                for instance in candidate.get("instances", [])
            }
            for file_row in files_by_patient.get(patient_id, []):
                if file_row.get("path") in selected_paths:
                    file_row["selection_status"] = "selected"
                elif file_row.get("path") in candidate_paths:
                    file_row["selection_status"] = "not_selected"
                    file_row["selection_reason"] = "eligible_candidate_not_chosen_by_manifest_policy"
                elif file_row.get("eligibility_status") == "ineligible":
                    file_row["selection_status"] = "skipped"
                    file_row["selection_reason"] = file_row.get("eligibility_reason")
            entry.update(
                status=selection.get("status"),
                requested_policy=requested_policy,
                policy=selection.get("policy"),
                fallback_reason=selection.get("fallback_reason"),
                candidates={
                    "cts": _candidate_rows(cts),
                    "frontals": _candidate_rows(frontals),
                    "laterals": _candidate_rows(laterals),
                },
                selected={
                    "ct": _manifest_candidate(selected_ct),
                    "frontal": _manifest_candidate(selected_frontal),
                    "lateral": _manifest_candidate(selected_lateral),
                },
                intervals=selection.get("intervals", {}),
            )
            if is_real:
                entry["respiratory_evidence"] = audit_selected_exams(entry["selected"])
            if selected_ct:
                entry["status"] = selection.get("status", "selected")
            else:
                entry["status"] = "no_valid_ct"
        if entry["annotation_conflict"]:
            entry.setdefault("warnings", []).append("patient_selection_annotation_conflict")
        entry["selection_fingerprint"] = _selection_fingerprint(entry)
        for destination in entry["destinations"]:
            entry["selection_by_destination"][destination] = {
                "policy": entry.get("policy", "largest_valid"),
                "status": entry["status"],
            }
        patients.append(entry)
        audit.add_patient(**{key: value for key, value in entry.items() if key != "candidates" and key != "selected"})

    cohort.verify_unchanged()
    audit.write(audit_dir / "dicom_audit.json")
    blockers = []
    for entry in patients:
        if not entry.get("selected", {}).get("ct"):
            blockers.append({"patient_id": entry["patient_id"], "reason": "no_valid_ct_candidate"})
        if any(name.startswith("real_") for name in entry["memberships"]):
            for role in ("frontal", "lateral"):
                if not entry.get("selected", {}).get(role):
                    blockers.append({"patient_id": entry["patient_id"], "reason": f"missing_required_real_{role}"})
        if entry.get("annotation_conflict"):
            blockers.append({"patient_id": entry["patient_id"], "reason": "patient_selection_annotation_conflict_requires_review"})
    manifest = {
        "schema_version": 1,
        "processing_contract_version": PROCESSING_CONTRACT_VERSION,
        "splits_dir": str(Path(args.splits_dir).resolve()),
        "split_hashes": {name: cohort.splits[name].sha256 for name in cohort.splits},
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "selection_default": args.ct_selection or "temporal_for_real_largest_valid_for_synthetic_only",
        "ready_for_generate": not blockers,
        "prepare_blockers": blockers,
        "patients": patients,
    }
    write_json(run_dir / "selection_manifest.json", manifest)
    return manifest


def generate(args) -> None:
    manifest_path = Path(args.manifest)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("processing_contract_version") != PROCESSING_CONTRACT_VERSION:
        raise ValueError("Manifest uses an older processing contract; prepare must be repeated before generation")
    if not args.approve_manifest:
        raise ValueError("Generation requires --approve-manifest after reviewing the prepared selection manifest")
    if not manifest.get("ready_for_generate"):
        raise ValueError(f"Manifest is not ready for generation: {manifest.get('prepare_blockers', [])}")
    cohort = Cohort(manifest["splits_dir"])
    current_hashes = {name: cohort.splits[name].sha256 for name in cohort.splits}
    if current_hashes != manifest.get("split_hashes"):
        raise ValueError("Split files differ from the hashes captured during prepare; generation blocked")
    entries = {item["patient_id"]: item for item in manifest["patients"]}
    if len(entries) != len(manifest["patients"]) or set(entries) != set(cohort.all_ids):
        raise ValueError("Selection manifest does not contain exactly the authorized cohort")
    for patient_id, entry in entries.items():
        if entry.get("memberships") != list(cohort.memberships[patient_id]):
            raise ValueError(f"Manifest memberships differ from splits for {patient_id}")
        if entry.get("selection_fingerprint") != _selection_fingerprint(entry):
            raise ValueError(f"Manifest selection changed after preparation for {patient_id}")

    output_root = Path(manifest["output_root"])
    logger = setup_logger("preprocess-generate", experiment_name="generate", filename_comment="generate")
    failures = []
    patient_results = {}
    journal = EventJournal(output_root / "audit" / "processing_events.jsonl")
    for patient_id in cohort.all_ids:
        entry = entries[patient_id]
        results = process_patient(entry, cohort, output_root, journal, logger)
        patient_results[patient_id] = results
        failures.extend({"patient_id": patient_id, "destination": destination, **result["error"]}
                        for destination, result in results.items() if result["status"] == "processing_failed")
    try:
        cohort.verify_unchanged()
        final_hashes = {name: hashlib.sha256((cohort.splits_dir / name).read_bytes()).hexdigest() for name in cohort.splits}
        split_integrity = {
            "split_hashes_unchanged": True,
            "initial_split_hashes": manifest["split_hashes"],
            "final_split_hashes": final_hashes,
        }
    except Exception as exc:
        split_integrity = {
            "split_hashes_unchanged": False,
            "initial_split_hashes": manifest.get("split_hashes", {}),
            "split_integrity_error": {"type": type(exc).__name__, "message": str(exc)},
        }
        failures.append({"reason": "protected_split_changed_during_generation", **split_integrity})
    audit_path = output_root / "audit" / "dicom_audit.json"
    update_processing_results(audit_path, patient_results, cohort)
    journal.consolidate(output_root / "audit" / "processing_events.json")
    coverage = validate_split_outputs(
        cohort,
        {split_name: destination_root(output_root, split_name.removesuffix(".txt")) for split_name in cohort.splits},
    )
    coverage.update(split_integrity)
    coverage["success"] = coverage["success"] and split_integrity["split_hashes_unchanged"] and not failures
    if not coverage["success"]:
        failures.append({"reason": "expected_written_coverage_mismatch", "coverage": coverage})
    coverage["processing_failures"] = [failure for failure in failures if failure.get("reason") != "expected_written_coverage_mismatch"]
    write_json(output_root / "audit" / "completion_report.json", coverage)
    if failures:
        raise RuntimeError(f"Generation incomplete for {len(failures)} patient(s); see audit/processing_events.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "generate"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-root", type=Path, default=Path("output/preprocess/GAN-dataset-complete"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approve-manifest", action="store_true", help="Confirm manual review of the complete prepare manifest before pixel/DRR processing")
    parser.add_argument("--dataset", action="append", help="Dataset directory under data-root; repeatable")
    parser.add_argument("--ct-selection", choices=("temporal", "largest_valid"))
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        prepare(args)
    else:
        if args.manifest is None:
            parser.error("--manifest is required for --stage generate")
        generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
