"""Cheap DICOM header inventory, independent of image processing classes."""

from collections import defaultdict
from pathlib import Path
import csv



def read_authorized_index(index_root, cohort, annotations, audit):
    """Resolve split membership and paths from the collection index before DICOM I/O."""
    index_path = Path(index_root) / "metadata.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Path-first inventory requires collection index {index_path}")
    grouped = defaultdict(lambda: {"metadata": None, "directories": []})
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        for row_number, record in enumerate(csv.DictReader(stream), 2):
            patient_id, status = cohort.resolve_patient_id(record.get("Subject ID"))
            if status != "matched":
                audit.add_series(
                    patient_id_raw=record.get("Subject ID"),
                    series_instance_uid=record.get("Series UID") or None,
                    study_instance_uid=record.get("Study UID") or None,
                    modality=record.get("Modality"),
                    file_location=record.get("File Location"),
                    file_count_indexed=record.get("Number of Images"),
                    index_path=str(index_path), index_row=row_number,
                    eligibility_status="skipped", selection_status="skipped",
                    reason="outside_all_splits" if status == "outside_all_splits" else status,
                )
                continue
            annotation = annotations.get(patient_id)
            if annotation is not None and annotation != "selected":
                audit.add_series(
                    patient_id=patient_id, memberships=list(cohort.memberships[patient_id]),
                    series_instance_uid=record.get("Series UID"),
                    file_location=record.get("File Location"),
                    eligibility_status="blocked", reason="patient_selection_annotation_conflict",
                    index_path=str(index_path), index_row=row_number,
                )
                continue
            modality = str(record.get("Modality", "")).strip().upper()
            if modality not in {"CT", "CR", "DX"}:
                audit.add_series(
                    patient_id=patient_id, memberships=list(cohort.memberships[patient_id]),
                    series_instance_uid=record.get("Series UID") or None,
                    study_instance_uid=record.get("Study UID") or None,
                    modality=modality or None, file_location=record.get("File Location"),
                    file_count_indexed=record.get("Number of Images"),
                    index_path=str(index_path), index_row=row_number,
                    eligibility_status="skipped", selection_status="skipped",
                    reason="unsupported_modality",
                )
                continue
            try:
                indexed_count = int(record.get("Number of Images") or 0)
            except (TypeError, ValueError):
                indexed_count = None
            location = record.get("File Location", "").strip()
            candidate = (Path(index_root) / location).resolve()
            root = Path(index_root).resolve()
            if not location or not candidate.is_relative_to(root) or not candidate.is_dir():
                audit.add_series(
                    patient_id=patient_id, memberships=list(cohort.memberships[patient_id]),
                    series_instance_uid=record.get("Series UID"),
                    file_location=location or None,
                    eligibility_status="blocked", reason="indexed_path_missing_or_outside_root",
                    index_path=str(index_path), index_row=row_number,
                )
                continue
            series_uid = str(record.get("Series UID") or "").strip() or None
            study_uid = str(record.get("Study UID") or "").strip() or None
            group_key = (patient_id, study_uid, series_uid or str(candidate), modality)
            group = grouped[group_key]
            group["metadata"] = {
                "patient_id": patient_id,
                "memberships": list(cohort.memberships[patient_id]),
                "study_instance_uid": study_uid,
                "series_instance_uid": series_uid,
                "modality": modality,
                "number_of_images_indexed": indexed_count,
                "index_path": str(index_path),
            }
            if candidate not in group["directories"]:
                group["directories"].append(candidate)

    series_rows = []
    for (patient_id, study_uid, series_uid, modality), group in sorted(grouped.items(), key=lambda item: tuple(str(part or "") for part in item[0])):
        metadata = group["metadata"]
        paths, seen = [], set()
        for directory in group["directories"]:
            # Retain the existing directory reader's extension and glob order;
            # CT pixel loading still performs its established z-position sort.
            for extension in (".dcm", ".dicom"):
                for path in directory.glob(f"*{extension}"):
                    if path.is_file() and path not in seen:
                        seen.add(path)
                        paths.append(path)
        rows = []
        for path in sorted(paths):
            row = audit.add_file(
                path=str(path.resolve()), relative_path=str(path.relative_to(index_root)),
                patient_id=patient_id, patient_id_raw=patient_id,
                memberships=metadata["memberships"], in_any_split=True,
                in_real=any(name.startswith("real_") for name in metadata["memberships"]),
                modality=modality, study_instance_uid=study_uid,
                series_instance_uid=series_uid, sop_instance_uid=None,
                read_status="indexed", index_path=str(index_path),
                number_of_images_indexed=metadata["number_of_images_indexed"],
            )
            rows.append(row)
        series_rows.append({"metadata": metadata, "paths": paths, "file_rows": rows})
    return series_rows


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
