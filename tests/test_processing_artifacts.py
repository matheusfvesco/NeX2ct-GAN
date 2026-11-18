import csv
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from lib.data.audit import DicomAudit, _processing_status, write_exam_rows, write_json
from lib.data.cohort import Cohort, SPLIT_FILES


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "paper-figure-scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_splits(root, values):
    root.mkdir()
    for name in SPLIT_FILES:
        ids = values.get(name, [])
        (root / name).write_text("".join(f"{patient}\n" for patient in ids), encoding="utf-8")
    return Cohort(root)


def test_clinical_json_csv_reconcile_incomplete_and_unreadable_patients(tmp_path):
    exporter = load_script("export-clinical-audit.py")
    cohort = write_splits(tmp_path / "splits", {"real_train.txt": ["a"], "real_test.txt": ["b"]})
    patients = [{"patient_id": patient, "memberships": list(cohort.memberships[patient]),
                 "status": "no_valid_ct", "selected": {}, "intervals": {}} for patient in cohort.real_ids]
    manifest = {"schema_version": 1, "splits_dir": str(cohort.splits_dir),
                "split_hashes": {name: split.sha256 for name, split in cohort.splits.items()}, "patients": patients}
    manifest_path, audit_path, events_path = [tmp_path / name for name in ("manifest.json", "audit.json", "events.json")]
    write_json(manifest_path, manifest)
    write_json(audit_path, {"schema_version": 1, "patients": []})
    write_json(events_path, {"schema_version": 1, "events": []})
    root = tmp_path / "hdf5"
    broken = root / "real" / "real_train" / "a" / "ct_xray_data.h5"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a valid HDF5")
    rows = exporter.export_clinical_audit(manifest_path, audit_path, events_path, tmp_path / "clinical", hdf5_root=root)
    assert [row["patient_id"] for row in rows] == ["a", "b"]
    assert rows[0]["hdf5_status"]["status"] == "unreadable"
    assert rows[1]["hdf5_status"]["status"] == "missing"
    assert all(row["intervention"]["status"] == "not_assessed" for row in rows)
    with (tmp_path / "clinical" / "clinical_pairs.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(cohort.real_ids)
    assert json.loads(csv_rows[0]["hdf5_status"])["status"] == "unreadable"
    run_path = tmp_path / "run.json"
    write_json(run_path, {"schema_version": 1, "selection": manifest, "patients": [], "events": []})
    consolidated = exporter.export_clinical_audit(
        run_path, run_path, run_path, tmp_path / "clinical-consolidated", hdf5_root=root)
    assert consolidated == rows
    subset_manifest = {**manifest, "patients": patients[:1], "subset": {"count_per_dataset": 1},
                       "active_patient_ids": ["a"]}
    write_json(run_path, {"schema_version": 1, "selection": subset_manifest, "patients": [], "events": []})
    subset_rows = exporter.export_clinical_audit(
        run_path, run_path, run_path, tmp_path / "clinical-subset", hdf5_root=root)
    assert [row["patient_id"] for row in subset_rows] == ["a"]
    summary = json.loads((tmp_path / "clinical-subset" / "clinical_summary.json").read_text())
    assert summary["execution_scope"] == "subset"
    cohort.verify_unchanged()

    pair_patients = []
    for index, patient_id in enumerate(cohort.real_ids):
        pair_patients.append({
            "patient_id": patient_id,
            "memberships": list(cohort.memberships[patient_id]),
            "selected": {"ct": {"path": "ct"}, "frontal": {"path": "frontal"},
                         "lateral": {"path": "lateral"}},
            "intervals": {
                "frontal_minus_ct": {"status": "available", "value": index + 1,
                                     "unit": "day", "precision": "day"},
                "lateral_minus_ct": {"status": "available" if index else "unavailable",
                                    "value": 3 if index else None, "unit": "day", "precision": "day"},
            },
        })
    pair_manifest = {**manifest, "patients": pair_patients}
    write_json(manifest_path, pair_manifest)
    exporter.export_clinical_audit(
        manifest_path, audit_path, events_path, tmp_path / "clinical-intervals")
    summary = json.loads((tmp_path / "clinical-intervals" / "clinical_summary.json").read_text())
    # A supplied numeric delta is not evidence that the acquisitions are comparable.
    assert summary["temporal_intervals"]["frontal_minus_ct"]["usable_proportion"] == 0
    assert summary["temporal_intervals"]["lateral_minus_ct"]["usable_count"] == 0
    assert summary["temporal_intervals"]["both_views"]["denominator"] == 2
    assert summary["temporal_conclusion"]["status"] == "inconclusive"

    empty_cohort = write_splits(tmp_path / "empty-splits", {})
    empty_manifest = {
        "schema_version": 1,
        "splits_dir": str(empty_cohort.splits_dir),
        "split_hashes": {name: split.sha256 for name, split in empty_cohort.splits.items()},
        "patients": [],
    }
    empty_manifest_path = tmp_path / "empty-manifest.json"
    write_json(empty_manifest_path, empty_manifest)
    exporter.export_clinical_audit(
        empty_manifest_path, audit_path, events_path, tmp_path / "clinical-empty")
    empty_summary = json.loads((tmp_path / "clinical-empty" / "clinical_summary.json").read_text())
    assert empty_summary["temporal_intervals"]["both_views"]["denominator"] == 0
    assert empty_summary["temporal_intervals"]["both_views"]["usable_proportion"] is None


def test_exam_audit_is_series_list_with_flat_identifiers_and_file_details(tmp_path):
    audit = DicomAudit()
    file_row = audit.add_file(path="/data/source/a.dcm", patient_id="a", dataset="source",
                              expected_patient=True, series_instance_uid="series-a", read_status="read_failed",
                              errors=[{"stage": "header", "message": "bad file"}])
    audit.add_series(path="/data/source", patient_id="a", dataset="source", expected_patient=True,
                     series_instance_uid="series-a", file_paths=[file_row["path"]],
                     eligibility_status="ineligible", reason="directory_classification_failed")
    output = tmp_path / "exams.json"
    write_exam_rows(audit, output)
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["patient_id"] == "a"
    assert rows[0]["series_instance_uid"] == "series-a"
    assert rows[0]["expected_patient"] is True
    assert rows[0]["should_include"] is False
    assert rows[0]["final_status"] == "failed"
    assert rows[0]["error_message"] == "bad file"
    assert rows[0]["files"][0]["path"] == "/data/source/a.dcm"
    assert rows[0]["files"][0]["errors"][0]["message"] == "bad file"


def test_run_audit_exports_observed_metadata_and_selection_to_csv(tmp_path):
    audit = DicomAudit()
    source = {
        "path": "/raw/patient/ct/slice.dcm",
        "study_instance_uid": "study-1",
        "series_instance_uid": "series-1",
        "sop_instance_uid": "sop-1",
        "temporal": {"raw": "20240101120000", "source": "AcquisitionDateTime",
                      "precision": "second",
                      "status": "valid", "raw_fields": {"AcquisitionDateTime": "20240101120000"}},
        "metadata": {"AcquisitionDateTime": "20240101120000"},
        "field_status": {"AcquisitionDateTime": "observed"},
    }
    audit.add_patient(patient_id="patient", selection_status="selected")
    audit.add_series(patient_id="patient", path="/raw/patient/ct", file_paths=[source["path"]],
                     study_instance_uid="study-1", series_instance_uid="series-1",
                     selection_status="selected", eligibility_status="eligible")
    audit.add_file(path=source["path"], patient_id="patient", selection_status="selected",
                   metadata_status="observed", source_metadata=source)
    audit.add_file(path="/raw/patient/not-selected.dcm", patient_id="patient",
                   selection_status="not_selected", metadata_status="not_verified")
    audit.add_file(path="/raw/patient/excluded.dcm", patient_id="patient",
                   selection_status="excluded", selection_reason="annotation_excluded",
                   metadata_status="not_verified")
    audit.add_file(path="/raw/patient/failed.dcm", patient_id="patient",
                   selection_status="selected", processing_status="processing_failed",
                   metadata_status="not_verified", errors=[{"message": "write failed"}])

    audit.persist_run_report({"run_id": "run-1", "status": "succeeded",
                              "inventory_complete": True, "selection": {"patients": []}}, tmp_path)

    with (tmp_path / "exams.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["selection_status"] for row in rows} == {"selected", "not_selected", "excluded"}
    selected = next(row for row in rows if row.get("source_metadata"))
    assert "20240101120000" in selected["source_metadata"]
    assert selected["AcquisitionDateTime"] == "20240101120000"
    assert selected["StudyInstanceUID"] == "study-1"
    assert (tmp_path / "run.json").is_file()
    assert (tmp_path / "exams.json").is_file()
    report = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert report["metadata_coverage"][0]["metadata_status_counts"]["observed"] == 1
    assert report["metadata_coverage"][0]["field_status_counts"]["AcquisitionDateTime"]["not_verified"] == 3


def test_interrupted_inventory_does_not_claim_unverified_patients_are_missing(tmp_path):
    audit = DicomAudit()
    report = {"status": "failed", "inventory_complete": False,
              "cohort_snapshot": {"memberships": [{"patient_id": "not-seen"}]}}
    audit.persist_run_report(report, tmp_path)
    coverage = json.loads((tmp_path / "run.json").read_text())["patient_coverage"]
    assert coverage["missing_ids"] == []
    assert coverage["not_yet_verified_ids"] == ["not-seen"]


def test_missing_required_patient_exam_fails_immediately_unless_continue_is_set():
    import pytest

    entry = {"patient_id": "patient", "memberships": ["synthetic_train.txt"],
             "selected": {"ct": None}, "annotation_conflict": False}
    with pytest.raises(ValueError, match="missing required exams"):
        Cohort.require_selection(entry)
    Cohort.require_selection(entry, continue_on_error=True)


def test_persist_run_report_sorts_the_inventory_only_once(tmp_path, monkeypatch):
    audit = DicomAudit()
    to_dict = Mock(wraps=audit.to_dict)
    monkeypatch.setattr(audit, "to_dict", to_dict)

    audit.persist_run_report({"status": "running"}, tmp_path)

    to_dict.assert_called_once()
    assert (tmp_path / "exams.json").is_file()
    assert (tmp_path / "run.json").is_file()


def test_non_selected_series_never_inherits_patient_processing_success():
    audit = DicomAudit()
    audit.add_file(path="chosen", patient_id="a", selection_status="selected")
    chosen = audit.add_series(patient_id="a", file_paths=["chosen"], eligibility_status="eligible")
    rejected = audit.add_series(patient_id="a", file_paths=["other"], eligibility_status="eligible")
    results = {"a": {"synthetic_train": {"status": "processed"}}}
    audit.record_processing_results(results)
    assert chosen["selection_status"] == "selected"
    assert chosen["processing_status"] == "processed"
    assert rejected["selection_status"] == "not_selected"
    assert rejected["processing_status"] == "not_applicable"
    assert _processing_status({"patient_id": "a"}, {"a": {"one": {"status": "reused"}}}) == "reused"
    assert _processing_status({"patient_id": "a"}, {"a": {"one": {"status": "processed"},
                                                            "two": {"status": "processing_failed"}}}) == "partially_processed"


def test_patch_preserves_source_json_and_real_reader_keys(tmp_path):
    patcher = load_script("patch_dataset.py")
    source = tmp_path / "source.h5"
    payload = json.dumps({"schema_version": 1, "optional": None, "evidence": ["expiração"]}, ensure_ascii=False)
    with h5py.File(source, "w") as stored:
        stored.create_dataset("ct", data=np.zeros((2, 2, 2)))
        for key, value in (("xray1", 0), ("xray2", 0), ("real_xray1", 1), ("real_xray2", 2)):
            stored.create_dataset(key, data=np.full((2, 2), value))
        for key in ("metadata_json", "provenance_json"):
            stored.create_dataset(key, data=payload, dtype=h5py.string_dtype("utf-8"))
    original_bytes = source.read_bytes()
    target = tmp_path / "real_train" / "patient" / "ct_xray_data.h5"
    mixed = tmp_path / "mixed_train" / "patient" / "ct_xray_data.h5"
    assert patcher.process_file(source, target, mixed) == (True, True)
    with h5py.File(source, "r") as original, h5py.File(target, "r") as derived:
        assert "real_xray1" in derived and "real_xray2" in derived
        assert np.array_equal(derived["xray1"][()], original["real_xray1"][()])
        assert derived["metadata_json"][()] == original["metadata_json"][()]
        assert derived["provenance_json"][()] == original["provenance_json"][()]
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("failed_destination", ["real", "mixed"])
def test_patch_copy_failure_preserves_previous_destination_and_propagates(tmp_path, monkeypatch, failed_destination):
    patcher = load_script("patch_dataset.py")
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as stored:
        for key in ("ct", "xray1", "xray2", "real_xray1", "real_xray2"):
            stored.create_dataset(key, data=np.ones((2, 2)))
    source_bytes = source.read_bytes()
    real = tmp_path / "real" / "ct_xray_data.h5"
    mixed = tmp_path / "mixed" / "ct_xray_data.h5"
    failed = real if failed_destination == "real" else mixed
    failed.parent.mkdir()
    failed.write_bytes(b"previous product")
    original_copy = h5py.Group.copy

    def fail_alias(self, source_name, destination, *args, **kwargs):
        if Path(destination.filename).parent == failed.parent and kwargs.get("name") == "xray1":
            raise OSError("copy failed")
        return original_copy(self, source_name, destination, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "copy", fail_alias)
    with pytest.raises(OSError, match="copy failed"):
        patcher.process_file(source, real, mixed)
    assert failed.read_bytes() == b"previous product"
    assert source.read_bytes() == source_bytes
    assert not list(tmp_path.rglob(".patch_*.h5"))
