"""Path-only inventory from the collection index; no DICOM files are opened here."""

from collections import defaultdict
import hashlib
import io
from pathlib import Path
import csv

SUBSET_SEED = 42
SUBSET_ALGORITHM = "sha256-patient-ranking-v1"


def _raise_unless_continuing(continue_on_error, message):
    if not continue_on_error:
        raise ValueError(message)


def _record_subset_exclusion(cohort, collection, context, audit):
    selected = getattr(cohort, "subset_by_dataset", None)
    if selected is None or context["patient_id"] in selected.get(collection, set()):
        return False
    audit.add_series(**context, eligibility_status="not_evaluated",
                     selection_status="not_evaluated", scope_status="not_selected_by_subset",
                     reason="not_selected_by_subset", file_paths=[])
    return True


def _resolve_index_context(record, cohort, root, index_path, row_number):
    patient_id, status = cohort.resolve_patient_id(record.get("Subject ID"))
    location = str(record.get("File Location") or "").strip()
    collection = str(record.get("Collection") or "").strip()
    if not collection:
        raise ValueError(f"Collection is missing at {index_path}:{row_number}")
    return patient_id, status, {
        "patient_id": patient_id, "patient_id_raw": record.get("Subject ID"),
        "identity_status": status,
        "patient_id_candidate": str(record.get("Subject ID") or "").strip() or None,
        "expected_patient": status == "matched", "dataset": collection,
        "memberships": list(cohort.memberships.get(patient_id, ())),
        "study_instance_uid": str(record.get("Study UID") or "").strip() or None,
        "series_instance_uid": str(record.get("Series UID") or "").strip() or None,
        "modality": str(record.get("Modality") or "").strip().upper(),
        "file_location": location, "file_count_indexed": record.get("Number of Images"),
        "index_path": str(index_path), "index_row": row_number,
        "relative_path": location, "path": str((root / location).resolve()),
    }


def select_subset_by_dataset(data_root, dataset_names, cohort, count, seed=SUBSET_SEED):
    """Select up to count unique authorized patients per indexed collection deterministically."""
    patients_by_collection, details, index_hashes = defaultdict(set), [], {}
    for dataset_name in sorted(dataset_names):
        index_path = Path(data_root) / dataset_name / "metadata.csv"
        raw = index_path.read_bytes()
        index_hashes[dataset_name] = hashlib.sha256(raw).hexdigest()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        required = {"Collection", "Subject ID", "File Location", "Modality", "Series UID", "Study UID"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Collection index {index_path} lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, record in enumerate(reader, 2):
            collection = str(record.get("Collection") or "").strip()
            if not collection:
                raise ValueError(f"Collection is missing at {index_path}:{row_number}")
            patient_id, status = cohort.resolve_patient_id(record.get("Subject ID"))
            if status == "matched":
                patients_by_collection[collection].add(patient_id)

    selected_by_collection = {}
    for collection, patients in sorted(patients_by_collection.items()):
        available = sorted(patients)
        ranked = sorted(
            available,
            key=lambda patient_id: (
                hashlib.sha256(f"{SUBSET_ALGORITHM}\0{seed}\0{collection}\0{patient_id}".encode("utf-8")).digest(),
                patient_id,
            ),
        )
        selected = tuple(sorted(ranked[:count]))
        selected_by_collection[collection] = set(selected)
        details.append({
            "dataset": collection,
            "available_authorized_patients": len(available),
            "selected_count": len(selected),
            "selected_patient_ids": list(selected),
        })
    return selected_by_collection, details, index_hashes


def read_authorized_index(index_root, cohort, annotations, audit):
    """Resolve split membership and paths from the collection index before DICOM I/O."""
    root = Path(index_root).resolve()
    continue_on_error = getattr(cohort, "continue_on_error", False)
    index_path = root / "metadata.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Path-first inventory requires collection index {index_path}")
    expected_hash = getattr(cohort, "subset_index_hashes", {}).get(root.name)
    if expected_hash and hashlib.sha256(index_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Collection index changed during subset selection: {index_path}")
    grouped = defaultdict(lambda: {"metadata": None, "directories": []})
    directory_owners = {}
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Collection", "Subject ID", "File Location", "Modality", "Series UID", "Study UID"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Collection index {index_path} lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, record in enumerate(reader, 2):
            patient_id, status, context = _resolve_index_context(
                record, cohort, root, index_path, row_number)
            if _record_subset_exclusion(cohort, context["dataset"], context, audit):
                continue
            if status != "matched":
                series_row = audit.add_series(**context, eligibility_status="skipped", selection_status="skipped", reason=status)
                series_row["file_paths"] = []
                if status == "patient_id_unresolved":
                    raise ValueError(f"Unresolved patient identity at {index_path}:{row_number}")
                _raise_unless_continuing(
                    continue_on_error,
                    f"Patient {status}: {context['patient_id_raw']!r} at {index_path}:{row_number}",
                )
                continue
            annotation = annotations.get(patient_id)
            if patient_id in annotations and annotation != "selected":
                audit.add_series(**context, eligibility_status="blocked", reason="patient_selection_annotation_conflict")
                _raise_unless_continuing(
                    continue_on_error, f"Patient selection annotation conflicts for {patient_id}"
                )
                continue
            modality = context["modality"]
            if modality not in {"CT", "CR", "DX"}:
                audit.add_series(**context, eligibility_status="skipped", selection_status="skipped", reason="unsupported_modality")
                continue
            if modality != "CT" and not any(name.startswith("real_") for name in context["memberships"]):
                audit.add_series(**context, eligibility_status="not_applicable", selection_status="skipped", reason="patient_not_in_real_cohort")
                continue
            try:
                raw_count = record.get("Number of Images")
                indexed_count = int(raw_count) if raw_count not in (None, "") else None
            except (TypeError, ValueError):
                indexed_count = None
            location = context["file_location"]
            candidate = (root / location).resolve()
            if not location or not candidate.is_relative_to(root) or not candidate.is_dir():
                audit.add_series(**context, eligibility_status="blocked", reason="indexed_path_missing_or_outside_root")
                _raise_unless_continuing(
                    continue_on_error,
                    f"Indexed series path is missing or outside dataset root: {location!r}",
                )
                continue
            series_uid = context["series_instance_uid"]
            study_uid = context["study_instance_uid"]
            group_key = (patient_id, study_uid, series_uid or str(candidate), modality)
            if candidate in directory_owners and directory_owners[candidate] != group_key:
                audit.add_series(**context, eligibility_status="blocked", reason="conflicting_series_directory_owner")
                raise ValueError(f"Conflicting collection index rows for the same series directory: {candidate}")
            directory_owners[candidate] = group_key
            group = grouped[group_key]
            if group["metadata"] is not None and group["metadata"]["dataset"] != context["dataset"]:
                raise ValueError(f"Conflicting Collection for series {series_uid!r} at {index_path}:{row_number}")
            if group["metadata"] is None:
                group["metadata"] = {
                    "patient_id": patient_id,
                    "dataset": context["dataset"],
                    "patient_id_raw": context["patient_id_raw"],
                    "memberships": list(cohort.memberships[patient_id]),
                    "study_instance_uid": study_uid,
                    "series_instance_uid": series_uid,
                    "modality": modality,
                    "number_of_images_indexed": indexed_count,
                    "index_path": str(index_path),
                }
            elif candidate not in group["directories"]:
                previous = group["metadata"]["number_of_images_indexed"]
                group["metadata"]["number_of_images_indexed"] = previous + indexed_count if previous is not None and indexed_count is not None else None
            if candidate not in group["directories"]:
                group["directories"].append(candidate)

    for _, group in sorted(grouped.items(), key=lambda item: tuple(str(part or "") for part in item[0])):
        yield _enumerate_indexed_series(group, root, audit)


def _enumerate_indexed_series(group, root, audit):
    """Enumerate only an approved group, retaining the directory reader's order."""
    metadata = group["metadata"]
    paths, seen = [], set()
    for directory in group["directories"]:
        for extension in (".dcm", ".dicom"):
            for path in directory.glob(f"*{extension}"):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    paths.append(path)
    rows = []
    for path in sorted(paths):
        rows.append(audit.add_file(
            path=str(path), relative_path=str(path.relative_to(root)),
            dataset=metadata["dataset"], expected_patient=True,
            patient_id=metadata["patient_id"], patient_id_raw=metadata["patient_id_raw"],
            patient_id_source="metadata.csv:Subject ID",
            memberships=metadata["memberships"], in_any_split=True,
            in_real=any(name.startswith("real_") for name in metadata["memberships"]),
            modality=metadata["modality"], study_instance_uid=metadata["study_instance_uid"],
            series_instance_uid=metadata["series_instance_uid"], sop_instance_uid=None,
            read_status="indexed", index_path=metadata["index_path"],
            number_of_images_indexed=metadata["number_of_images_indexed"],
        ))
    return {"metadata": metadata, "paths": paths, "file_rows": rows}


def record_directory_result(audit, rows, directory_path, patient_id, *, directory=None, error=None):
    """Record classification locally, without rescanning every previously seen file."""
    eligible = directory is not None and (not directory.is_ct or directory.is_valid_axial_scan)
    reason = "eligible_series" if eligible else "series_failed_technical_criteria"
    if error is not None:
        reason = "directory_classification_failed"
    for row in rows:
        row.update(eligibility_status="eligible" if eligible else "ineligible", eligibility_reason=reason)
        if not eligible:
            row.update(selection_status="skipped", selection_reason=reason, processing_status="skipped")
        if error is not None:
            row["errors"].append({"stage": "directory_classification", "code": reason,
                                  "exception_type": type(error).__name__, "message": str(error)})
    audit.add_series(
        path=str(directory_path), patient_id=patient_id,
        expected_patient=bool(patient_id), dataset=rows[0].get("dataset") if rows else None,
        file_paths=[row["path"] for row in rows], file_count=len(rows),
        series_instance_uid=next((row.get("series_instance_uid") for row in rows), None),
        study_instance_uid=next((row.get("study_instance_uid") for row in rows), None),
        modality=rows[0].get("modality") if rows else None,
        source_directories=sorted({str(Path(row["path"]).parent) for row in rows}),
        eligibility_status="eligible" if eligible else "ineligible", reason=reason,
    )


def annotated_view_paths(paths, classifier):
    """Retain positive annotations and at most one genuinely unannotated fallback."""
    selected, missing, roles = set(), [], set()
    for path in paths:
        try:
            role = classifier.classify(path)
        except KeyError:
            missing.append(path)
            continue
        if role in {"frontal", "lateral"}:
            selected.add(path)
            roles.add(role)
    if roles != {"frontal", "lateral"} and missing:
        selected.add(missing[len(missing) // 2])
    return [path for path in paths if path in selected]
