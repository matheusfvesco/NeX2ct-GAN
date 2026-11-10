"""Path-only inventory from the collection index; no DICOM files are opened here."""

from collections import defaultdict
from pathlib import Path
import csv


def read_authorized_index(index_root, cohort, annotations, audit):
    """Resolve split membership and paths from the collection index before DICOM I/O."""
    root = Path(index_root).resolve()
    index_path = root / "metadata.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Path-first inventory requires collection index {index_path}")
    grouped = defaultdict(lambda: {"metadata": None, "directories": []})
    directory_owners = {}
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Subject ID", "File Location", "Modality", "Series UID", "Study UID"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Collection index {index_path} lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        for row_number, record in enumerate(reader, 2):
            patient_id, status = cohort.resolve_patient_id(record.get("Subject ID"))
            context = {
                "patient_id": patient_id, "patient_id_raw": record.get("Subject ID"),
                "memberships": list(cohort.memberships.get(patient_id, ())),
                "study_instance_uid": str(record.get("Study UID") or "").strip() or None,
                "series_instance_uid": str(record.get("Series UID") or "").strip() or None,
                "modality": str(record.get("Modality") or "").strip().upper(),
                "file_location": str(record.get("File Location") or "").strip(),
                "file_count_indexed": record.get("Number of Images"),
                "index_path": str(index_path), "index_row": row_number,
            }
            if status != "matched":
                audit.add_series(**context, eligibility_status="skipped", selection_status="skipped", reason=status)
                continue
            annotation = annotations.get(patient_id)
            if patient_id in annotations and annotation != "selected":
                audit.add_series(**context, eligibility_status="blocked", reason="patient_selection_annotation_conflict")
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
                continue
            series_uid = context["series_instance_uid"]
            study_uid = context["study_instance_uid"]
            group_key = (patient_id, study_uid, series_uid or str(candidate), modality)
            if candidate in directory_owners and directory_owners[candidate] != group_key:
                raise ValueError(f"Conflicting collection index rows for the same series directory: {candidate}")
            directory_owners[candidate] = group_key
            group = grouped[group_key]
            if group["metadata"] is None:
                group["metadata"] = {
                    "patient_id": patient_id,
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
