"""Prepare split-authorized DICOM selections and generate their HDF5 products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from lib.data.audit import DicomAudit, EventJournal, write_exam_rows, write_json
from lib.data.cohort import Cohort
from lib.data.dicom import DicomDataset
from lib.data.selection import select_exams, selected_intervals
from lib.data.respiratory import audit_selected_exams
from lib.data.validation import validate_split_outputs
from lib.data.generation import PROCESSING_CONTRACT_VERSION, destination_root, process_patient
from lib.data.inventory import SUBSET_ALGORITHM, SUBSET_SEED, select_subset_by_dataset
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
    header = getattr(value, "header", None)
    header_path = paths[len(paths) // 2] if role == "ct" else path
    if header is not None and str(header_path) in audit_by_path:
        audit_by_path[str(header_path)].update(
            sop_instance_uid=header.get("SOPInstanceUID"), study_instance_uid=header.get("StudyInstanceUID"),
            series_instance_uid=header.get("SeriesInstanceUID"), read_status="read_ok")
    for instance_path in paths:
        row = audit_by_path.get(str(instance_path), {})
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
    eligible = value.is_valid_axial_scan if role == "ct" else value.is_valid
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
            row = rows.get(str(path))
            try:
                view = directory.cache.get("views_by_path", {}).get(path)
                if view is None:
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
                if not getattr(audit, "continue_on_error", False):
                    raise
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


def _ensure_subset(data_root: Path, datasets: list[str] | None, cohort: Cohort, count: int | None) -> dict | None:
    if count is None:
        return None
    datasets = datasets if datasets is not None else _discover_datasets(data_root, None)
    if cohort.subset_by_dataset is None:
        selected, details, index_hashes = select_subset_by_dataset(data_root, datasets, cohort, count)
        cohort.subset_by_dataset = selected
        cohort.subset_index_hashes = index_hashes
        cohort.subset_details = details
    else:
        selected = cohort.subset_by_dataset
        details = cohort.subset_details
        index_hashes = cohort.subset_index_hashes
    cohort.active_patient_ids = set().union(*selected.values()) if selected else set()
    return {
        "count_per_dataset": count,
        "seed": SUBSET_SEED,
        "algorithm": SUBSET_ALGORITHM,
        "datasets": details,
        "selected_by_dataset": {name: sorted(ids) for name, ids in selected.items()},
        "active_patient_ids": sorted(cohort.active_patient_ids),
        "metadata_index_sha256": index_hashes,
    }


def _ensure_patient_selection(entry, memberships, continue_on_error):
    if continue_on_error:
        return
    roles = ("ct", "frontal", "lateral") if any(
        name.startswith("real_") for name in memberships
    ) else ("ct",)
    missing = [role for role in roles if not entry.get("selected", {}).get(role)]
    if missing or entry["annotation_conflict"]:
        raise ValueError(f"Patient {entry['patient_id']}: missing required exams {missing}; "
                         f"annotation_conflict={entry['annotation_conflict']}")


def _persist_audits(audit: DicomAudit, report: dict, audit_dir: Path) -> None:
    payload = audit.to_dict()
    write_exam_rows(audit, audit_dir / "exams.json", report.get("run_id"))
    now = datetime.now(timezone.utc).isoformat()
    report["updated_at_utc"] = now
    if report.get("status") != "running":
        report["finished_at_utc"] = now
    report.update(series=payload["series"], patients=payload["patients"])
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
    write_json(audit_dir / "run.json", report)


def prepare(args, audit: DicomAudit | None = None, cohort: Cohort | None = None) -> dict:
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    cohort = cohort or Cohort(Path(args.splits_dir).resolve())
    cohort.continue_on_error = getattr(args, "continue_on_error", False)
    audit = audit or DicomAudit()
    audit.continue_on_error = cohort.continue_on_error
    logger = setup_logger("preprocess-prepare", experiment_name="prepare", filename_comment="prepare")
    datasets = _discover_datasets(data_root, args.dataset)
    if not datasets:
        raise ValueError(f"No dataset directories found under {data_root}")
    subset_report = _ensure_subset(data_root, datasets, cohort, getattr(args, "subset", None))
    empty_subset_datasets = [row["dataset"] for row in (subset_report or {}).get("datasets", [])
                             if row["selected_count"] == 0]
    if empty_subset_datasets and not cohort.continue_on_error:
        raise ValueError(f"No authorized patients available for subset in dataset(s): {empty_subset_datasets}")
    annotation_path = data_root / "patient-selection-annotation.json"
    with annotation_path.open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    if not isinstance(annotations, dict):
        raise ValueError(f"Expected an ID-to-selection JSON object in {annotation_path}")
    classifier = XRayJsonClassifier(root_data=data_root)
    classifier.load_json(root_data=data_root)
    dataset = DicomDataset(data_root, accepted_datasets=datasets, logger=logger, cohort=cohort, audit=audit)
    dataset.continue_on_error = cohort.continue_on_error
    dataset.patient_selection_annotations = annotations
    dataset.load()

    files_by_patient = {}
    for row in audit.files:
        files_by_patient.setdefault(row.get("patient_id"), []).append(row)
    patients = []
    for patient_id in cohort.processing_ids:
        membership = list(cohort.memberships[patient_id])
        entry = {
            "patient_id": patient_id,
            "memberships": membership,
            "destinations": list(cohort.destinations(patient_id)),
            "status": "no_valid_candidates",
            "selection_by_destination": {},
            "destination_status": {destination: "not_attempted" for destination in cohort.destinations(patient_id)},
            "processing_status": "not_attempted",
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
                entry["respiratory_evidence"] = audit_selected_exams(
                    {"ct": selected_ct, "frontal": selected_frontal, "lateral": selected_lateral})
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
        _ensure_patient_selection(entry, membership, cohort.continue_on_error)

    cohort.verify_unchanged()
    blockers = []
    blockers.extend({"reason": "no_authorized_patients_in_subset_dataset", "dataset": dataset_name}
                    for dataset_name in empty_subset_datasets)
    for row in audit.files:
        if row.get("errors"):
            blockers.append({"patient_id": row.get("patient_id"), "reason": "dicom_read_or_classification_failed",
                             "path": row.get("path"), "errors": row["errors"]})
    for row in audit.series:
        if row.get("errors") or row.get("reason") in {"outside_all_splits", "patient_id_unresolved", "patient_selection_annotation_conflict", "indexed_path_missing_or_outside_root", "directory_classification_failed"}:
            blockers.append({"patient_id": row.get("patient_id"), "reason": row.get("reason"), "path": row.get("file_location") or row.get("path")})
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
        "subset": subset_report,
        "active_patient_ids": list(cohort.processing_ids),
        "subset_index_sha256": getattr(cohort, "subset_index_hashes", {}),
    }
    return manifest


def _process_cohort(cohort, entries, output_root, logger, continue_on_error, blockers, checkpoint):
    failures = list(blockers) if continue_on_error else []
    patient_results = {}
    journal = EventJournal()
    journal.continue_on_error = continue_on_error
    for patient_id in cohort.processing_ids:
        cohort.verify_unchanged()
        blocked = any(item.get("patient_id") == patient_id for item in blockers)
        if blocked:
            patient_results[patient_id] = {
                destination: {"status": "not_attempted", "reason": "selection_blocked"}
                for destination in cohort.destinations(patient_id)
            }
            continue
        results = process_patient(entries[patient_id], cohort, output_root, journal, logger)
        patient_results[patient_id] = results
        failures.extend({"patient_id": patient_id, "destination": destination, **result["error"]}
                        for destination, result in results.items() if result["status"] == "processing_failed")
        checkpoint(patient_results, failures, journal.events)
        if failures and not continue_on_error:
            break
    return patient_results, failures, journal.consolidate(None)


def _record_file_results(audit, patient_results):
    for row in audit.files:
        if row.get("selection_status") != "selected":
            continue
        results = patient_results.get(row.get("patient_id"), {})
        row["destination_results"] = results
        row["processing_status"] = _destination_status(results)


def _destination_status(results):
    statuses = {item["status"] for item in results.values()}
    if not statuses or statuses == {"not_attempted"}:
        return "not_attempted"
    if statuses == {"reused"}:
        return "reused"
    if statuses <= {"processed", "reused"}:
        return "processed"
    if statuses & {"processed", "reused"}:
        return "partially_processed"
    return "processing_failed"


def _record_series_results(audit, patient_results):
    selected_paths = {row["path"] for row in audit.files if row.get("selection_status") == "selected"}
    for row in audit.series:
        if not selected_paths.intersection(row.get("file_paths", [])):
            row["processing_status"] = "not_applicable"
            if row.get("eligibility_status") == "eligible":
                row.update(selection_status="not_selected", reason="eligible_candidate_not_chosen")
            continue
        row["selection_status"] = "selected"
        results = patient_results.get(row.get("patient_id"))
        row["processing_status"] = _destination_status(results or {})
        if results is not None:
            row["destination_results"] = results


def _record_patient_results(audit, patient_results):
    for row in audit.patients:
        patient_id = row["patient_id"]
        results = patient_results.get(patient_id, {})
        row["destination_results"] = results
        row["destination_status"] = {
            destination: results.get(destination, {}).get("status", "not_attempted")
            for destination in row.get("destinations", [])
        }
        row["processing_status"] = _destination_status(results)


def _record_processing_results(audit, patient_results):
    _record_file_results(audit, patient_results)
    _record_series_results(audit, patient_results)
    _record_patient_results(audit, patient_results)


def _split_integrity(cohort, manifest):
    try:
        cohort.verify_unchanged()
        final_hashes = {name: hashlib.sha256((cohort.splits_dir / name).read_bytes()).hexdigest() for name in cohort.splits}
        return {"split_hashes_unchanged": True,
                "initial_split_hashes": manifest["split_hashes"], "final_split_hashes": final_hashes}, []
    except Exception as exc:
        result = {"split_hashes_unchanged": False,
                  "initial_split_hashes": manifest.get("split_hashes", {}),
                  "split_integrity_error": {"type": type(exc).__name__, "message": str(exc)}}
        return result, [{"reason": "protected_split_changed_during_generation", **result}]


def _restore_subset_scope(cohort, manifest):
    subset = manifest.get("subset")
    if subset is None:
        return
    cohort.active_patient_ids = set(manifest["active_patient_ids"])
    cohort.subset_by_dataset = {
        name: set(patient_ids) for name, patient_ids in subset["selected_by_dataset"].items()
    }
    selected_union = set().union(*cohort.subset_by_dataset.values()) if cohort.subset_by_dataset else set()
    if selected_union != cohort.active_patient_ids or not selected_union <= set(cohort.all_ids):
        raise ValueError("Subset scope differs from selected authorized patients")
    cohort.subset_index_hashes = manifest.get("subset_index_sha256", {})
    for dataset_name, expected_hash in cohort.subset_index_hashes.items():
        index_path = Path(manifest["data_root"]) / dataset_name / "metadata.csv"
        if hashlib.sha256(index_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Collection index changed after subset selection: {index_path}")


def generate(args, manifest: dict, audit: DicomAudit, run_report: dict) -> None:
    if manifest.get("processing_contract_version") != PROCESSING_CONTRACT_VERSION:
        raise ValueError("Manifest uses an older processing contract; prepare must be repeated before generation")
    continue_on_error = getattr(args, "continue_on_error", False)
    if not manifest.get("ready_for_generate") and not continue_on_error:
        raise ValueError(f"Manifest is not ready for generation: {manifest.get('prepare_blockers', [])}")
    cohort = Cohort(manifest["splits_dir"])
    cohort.continue_on_error = continue_on_error
    _restore_subset_scope(cohort, manifest)
    current_hashes = {name: cohort.splits[name].sha256 for name in cohort.splits}
    if current_hashes != manifest.get("split_hashes"):
        raise ValueError("Split files differ from the hashes captured during prepare; generation blocked")
    entries = {item["patient_id"]: item for item in manifest["patients"]}
    if len(entries) != len(manifest["patients"]) or set(entries) != set(cohort.processing_ids):
        raise ValueError("Selection manifest does not contain exactly the authorized cohort")
    for patient_id, entry in entries.items():
        if entry.get("memberships") != list(cohort.memberships[patient_id]):
            raise ValueError(f"Manifest memberships differ from splits for {patient_id}")
        if entry.get("selection_fingerprint") != _selection_fingerprint(entry):
            raise ValueError(f"Manifest selection changed after preparation for {patient_id}")

    output_root = Path(manifest["output_root"])
    logger = setup_logger("preprocess-generate", experiment_name="generate", filename_comment="generate")
    preexisting = validate_split_outputs(
        cohort, {name: destination_root(output_root, name.removesuffix(".txt")) for name in cohort.splits})
    run_report["preexisting_outputs"] = preexisting
    output_failures = [
        {"reason": "invalid_preexisting_outputs", "split": row["split"],
         "unexpected_ids": row["unexpected_ids"], "duplicate_outputs": row["duplicate_outputs"]}
        for row in preexisting["splits"] if row["unexpected_ids"] or row["duplicate_outputs"]
    ]
    if output_failures and not continue_on_error:
        raise ValueError(f"Unexpected or duplicate preexisting outputs: {output_failures}")
    def checkpoint(results, failures, events):
        run_report.update(patient_results=results, processing_failures=failures, events=list(events))
        _record_processing_results(audit, results)
        _persist_audits(audit, run_report, output_root / "audit")

    patient_results, failures, events = _process_cohort(
        cohort, entries, output_root, logger, continue_on_error, manifest.get("prepare_blockers", []), checkpoint)
    failures.extend(output_failures)
    run_report.update(events=events, patient_results=patient_results,
                      processing_failures=failures)
    split_integrity, split_failures = _split_integrity(cohort, manifest)
    failures.extend(split_failures)
    _record_processing_results(audit, patient_results)
    coverage = validate_split_outputs(
        cohort,
        {split_name: destination_root(output_root, split_name.removesuffix(".txt")) for split_name in cohort.splits},
    )
    coverage.update(split_integrity)
    coverage["success"] = coverage["success"] and split_integrity["split_hashes_unchanged"] and not failures
    if not coverage["success"]:
        failures.append({"reason": "expected_written_coverage_mismatch", "coverage": coverage})
    coverage["processing_failures"] = [failure for failure in failures if failure.get("reason") != "expected_written_coverage_mismatch"]
    run_report.update(status="failed" if failures or not coverage["success"] else "succeeded",
                      completion=coverage, processing_failures=failures)
    if failures:
        raise RuntimeError(f"Generation incomplete for {len(failures)} patient(s); see audit/run.json")


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--subset requires a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("--subset requires a positive integer")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-root", type=Path, default=Path("output/preprocess/GAN-dataset-complete"))
    parser.add_argument("--dataset", action="append", help="Dataset directory under data-root; repeatable")
    parser.add_argument("--ct-selection", choices=("temporal", "largest_valid"))
    parser.add_argument("--continue-on-error", action="store_true", help="Continue independent inventory/processing after recoverable errors; exit nonzero if any error occurred")
    parser.add_argument("--subset", nargs="?", const=3, default=None, type=positive_int, metavar="X",
                        help="Select up to X patients per dataset deterministically (default X=3)")
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root).resolve()
    if output_root == data_root or output_root.is_relative_to(data_root):
        parser.error("--output-root must be outside --data-root; raw input is read-only")
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit = DicomAudit()
    run_report = {
        "schema_version": 1, "run_id": uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "stage": "split_validation", "inventory_complete": False,
        "continue_on_error": args.continue_on_error,
        "configuration": {"data_root": str(Path(args.data_root).resolve()),
                           "splits_dir": str(Path(args.splits_dir).resolve()),
                           "output_root": str(output_root), "datasets": args.dataset,
                           "ct_selection": args.ct_selection or "temporal"},
    }

    try:
        cohort = Cohort(Path(args.splits_dir).resolve())
        cohort.continue_on_error = args.continue_on_error
        run_report["cohort_snapshot"] = cohort.snapshot()
        run_report["subset"] = _ensure_subset(
            data_root, None if args.dataset is None else _discover_datasets(data_root, args.dataset),
            cohort, args.subset)
        run_report["execution_scope"] = "subset" if args.subset is not None else "full"
        _persist_audits(audit, run_report, audit_dir)
        run_report["stage"] = "inventory"
        manifest = prepare(args, audit, cohort)
        run_report["selection"] = manifest
        run_report["inventory_complete"] = True
        run_report["stage"] = "generation"
        _persist_audits(audit, run_report, audit_dir)
        generate(args, manifest, audit, run_report)
        run_report.update(status="succeeded", stage="complete", exit_code=0)
        _persist_audits(audit, run_report, audit_dir)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        run_report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          exit_code=130 if isinstance(exc, KeyboardInterrupt) else 1,
                          fatal_error={"type": type(exc).__name__, "message": str(exc)})
        try:
            _persist_audits(audit, run_report, audit_dir)
        except Exception as audit_exc:
            parser.exit(1, f"Audit persistence failed: {type(audit_exc).__name__}: {audit_exc}\n")
        parser.exit(run_report["exit_code"], f"{type(exc).__name__}: {exc}\nAudit: {audit_dir / 'run.json'}\n")


if __name__ == "__main__":
    raise SystemExit(main())
