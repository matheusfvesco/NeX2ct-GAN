"""Expected-versus-written coverage and split-integrity reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import h5py

from lib.data.audit import write_json
from lib.data.cohort import Cohort


def validate_split_outputs(
    cohort: Cohort,
    output_roots: dict[str, str | Path],
    *,
    filename: str = "ct_xray_data.h5",
) -> dict:
    """Compare literal split IDs with HDF5 files under each corresponding destination."""
    missing_roots = sorted(set(cohort.splits) - set(output_roots))
    if missing_roots:
        raise ValueError(f"Missing output roots for splits: {missing_roots}")
    report = {"schema_version": 1, "splits": [], "success": True}
    for split_name in cohort.splits:
        root = Path(output_roots[split_name])
        expected_order = cohort.expected_ids(split_name)
        expected = set(expected_order)
        authorized = set(cohort.splits[split_name].ids)
        paths = sorted(root.rglob(filename)) if root.exists() else []
        by_id: dict[str, list[str]] = {}
        for path in paths:
            patient_id = path.parent.name
            by_id.setdefault(patient_id, []).append(str(path))
        written = set(by_id)
        duplicates_all = {key: values for key, values in sorted(by_id.items()) if len(values) > 1}
        out_of_scope = sorted((written & authorized) - expected)
        duplicates = {key: values for key, values in duplicates_all.items() if key not in out_of_scope}
        out_of_scope_duplicates = {key: values for key, values in duplicates_all.items() if key in out_of_scope}
        invalid = []
        for patient_id in sorted(written - set(out_of_scope)):
            for path in by_id[patient_id]:
                try:
                    _validate_output(path, patient_id, split_name)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    invalid.append({"patient_id": patient_id, "path": path,
                                    "exception_type": type(exc).__name__, "message": str(exc)})
        row = {
            "split": split_name,
            "output_root": str(root),
            "expected_ids": list(expected_order),
            "written_ids": sorted(written),
            "missing_ids": sorted(expected - written),
            "unexpected_ids": sorted(written - authorized),
            "out_of_scope_ids": out_of_scope,
            "duplicate_outputs": duplicates,
            "out_of_scope_duplicate_outputs": out_of_scope_duplicates,
            "invalid_outputs": invalid,
            "success": expected <= written and not duplicates and not invalid and not (written - authorized),
        }
        report["success"] = report["success"] and row["success"]
        report["splits"].append(row)
    return report


def _validate_output(path, patient_id, split_name):
    """Validate shape and provenance from HDF5 headers, without reading image arrays."""
    keys = {"ct": 3, "xray1": 2, "xray2": 2}
    if split_name.startswith("real_"):
        keys.update(real_xray1=2, real_xray2=2)
    with h5py.File(path, "r") as stored:
        for key, dimensions in keys.items():
            dataset = stored[key]
            if dataset.ndim != dimensions or any(size <= 0 for size in dataset.shape):
                raise ValueError(f"Invalid {key} shape: {dataset.shape}")
        metadata = json.loads(stored["metadata_json"].asstr()[()])
        provenance = json.loads(stored["provenance_json"].asstr()[()])
        if not isinstance(metadata, dict) or not isinstance(provenance, dict):
            raise ValueError("HDF5 metadata and provenance must be JSON objects")
        if metadata.get("schema_version") != 1 or provenance.get("schema_version") != 1:
            raise ValueError("Unsupported HDF5 metadata/provenance schema")
        if provenance.get("patient_id") != patient_id:
            raise ValueError("HDF5 patient ID differs from its directory")
        destination = stored.attrs.get("product_destination", provenance.get("destination"))
        if destination != split_name.removesuffix(".txt"):
            raise ValueError("HDF5 destination differs from the split")
        if split_name not in provenance.get("memberships", []):
            raise ValueError("HDF5 provenance does not authorize this split")


def validate_patient_record_coverage(expected_ids: list[str] | tuple[str, ...], records: list[dict], id_field: str = "patient_id") -> dict:
    expected = set(expected_ids)
    counts = Counter(str(record.get(id_field, "")) for record in records)
    actual = set(counts)
    duplicates = sorted(patient_id for patient_id, count in counts.items() if count != 1)
    return {
        "expected_ids": sorted(expected),
        "record_ids": sorted(actual),
        "missing_ids": sorted(expected - actual),
        "unexpected_ids": sorted(actual - expected),
        "duplicate_ids": duplicates,
        "success": expected == actual and not duplicates,
    }


def write_completion_report(cohort: Cohort, output_roots: dict[str, str | Path], path: str | Path) -> dict:
    report = validate_split_outputs(cohort, output_roots)
    try:
        cohort.verify_unchanged()
        report["split_hashes_unchanged"] = True
    except Exception as exc:
        report["split_hashes_unchanged"] = False
        report["split_integrity_error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["success"] = report["success"] and report["split_hashes_unchanged"]
    write_json(path, report)
    return report
