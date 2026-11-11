import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np

from lib.data.audit import DicomAudit, write_exam_rows, write_json
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


def test_main_runs_both_stages_and_writes_only_two_audit_jsons(tmp_path, monkeypatch):
    import preprocess

    splits = write_splits(tmp_path / "splits", {})
    args_seen = []
    monkeypatch.setattr(preprocess, "prepare", lambda args, audit, cohort: {
        "schema_version": 1, "processing_contract_version": preprocess.PROCESSING_CONTRACT_VERSION,
        "splits_dir": str(cohort.splits_dir), "split_hashes": {}, "output_root": str(tmp_path / "out"),
        "patients": [], "ready_for_generate": True, "prepare_blockers": [],
    })
    monkeypatch.setattr(preprocess, "generate", lambda args, manifest, audit, report: args_seen.append(args))
    assert preprocess.main(["--splits-dir", str(splits.splits_dir), "--output-root", str(tmp_path / "out")]) == 0
    audit_dir = tmp_path / "out" / "audit"
    assert sorted(path.name for path in audit_dir.glob("*.json")) == ["exams.json", "run.json"]
    report = json.loads((audit_dir / "run.json").read_text(encoding="utf-8"))
    assert report["status"] == "succeeded"
    assert len(args_seen) == 1


def test_subset_cli_accepts_bare_or_positive_count_and_caps_per_dataset(tmp_path, monkeypatch):
    import pytest
    import preprocess

    for args, expected_count, expected_size in (
        ([], None, 5),
        (["--subset"], 3, 3),
        (["--subset", "1"], 1, 1),
        (["--subset", "99"], 99, 5),
    ):
        data_root = tmp_path / f"data-{expected_count}"
        dataset_root = data_root / "dataset-a"
        dataset_root.mkdir(parents=True)
        (dataset_root / "metadata.csv").write_text(
            "Series UID,Subject ID,Study UID,Modality,File Location\n" +
            "".join(f"s-{index},p{index},study,CT,./missing-{index}\n" for index in range(5)),
            encoding="utf-8",
        )
        cohort = write_splits(tmp_path / f"splits-{expected_count}", {
            "synthetic_train.txt": [f"p{index}" for index in range(5)],
        })
        observed = {}
        def prepare(parsed, audit, active_cohort):
            observed["count"] = parsed.subset
            observed["active"] = active_cohort.processing_ids
            return {
                "schema_version": 1,
                "processing_contract_version": preprocess.PROCESSING_CONTRACT_VERSION,
                "splits_dir": str(active_cohort.splits_dir),
                "split_hashes": {name: split.sha256 for name, split in active_cohort.splits.items()},
                "output_root": str(tmp_path / f"out-{expected_count}"),
                "patients": [], "ready_for_generate": True, "prepare_blockers": [],
            }
        monkeypatch.setattr(preprocess, "prepare", prepare)
        monkeypatch.setattr(preprocess, "generate", lambda *unused: None)
        result = preprocess.main([
            "--data-root", str(data_root), "--splits-dir", str(cohort.splits_dir),
            "--output-root", str(tmp_path / f"out-{expected_count}"), *args,
        ])
        assert result == 0
        assert observed["count"] == expected_count
        assert len(observed["active"]) == expected_size

    for invalid in ("0", "-2", "1.5", "many"):
        with pytest.raises(SystemExit) as error:
            preprocess.main(["--subset", invalid])
        assert error.value.code == 2


def test_missing_required_patient_exam_fails_immediately_unless_continue_is_set():
    import pytest
    import preprocess

    entry = {"patient_id": "patient", "selected": {"ct": None}, "annotation_conflict": False}
    with pytest.raises(ValueError, match="missing required exams"):
        preprocess._ensure_patient_selection(entry, ("synthetic_train.txt",), False)
    preprocess._ensure_patient_selection(entry, ("synthetic_train.txt",), True)


def test_generate_restores_subset_instead_of_requiring_the_full_cohort(tmp_path, monkeypatch):
    import preprocess

    cohort = write_splits(tmp_path / "splits", {"synthetic_train.txt": ["a", "b", "c"]})
    data_root = tmp_path / "data"
    source = data_root / "source"
    source.mkdir(parents=True)
    (source / "metadata.csv").write_text(
        "Series UID,Subject ID,Study UID,Modality,File Location\n"
        "one,a,study,CT,./missing\ntwo,b,study,CT,./missing\nthree,c,study,CT,./missing\n",
        encoding="utf-8")
    subset = preprocess._ensure_subset(data_root, ["source"], cohort, 1)
    chosen = cohort.processing_ids[0]
    entry = {"patient_id": chosen, "memberships": list(cohort.memberships[chosen])}
    entry["selection_fingerprint"] = preprocess._selection_fingerprint(entry)
    manifest = {
        "processing_contract_version": preprocess.PROCESSING_CONTRACT_VERSION,
        "splits_dir": str(cohort.splits_dir), "data_root": str(data_root),
        "output_root": str(tmp_path / "output"), "ready_for_generate": True,
        "split_hashes": {name: split.sha256 for name, split in cohort.splits.items()},
        "subset": subset, "active_patient_ids": [chosen],
        "subset_index_sha256": subset["metadata_index_sha256"], "patients": [entry],
    }
    def validate(active_cohort, roots):
        assert active_cohort.processing_ids == (chosen,)
        assert active_cohort.expected_ids("synthetic_train.txt") == (chosen,)
        return {"success": True, "splits": []}
    def process(active_cohort, entries, *args):
        assert active_cohort.processing_ids == (chosen,)
        assert set(entries) == {chosen}
        return {}, [], []
    monkeypatch.setattr(preprocess, "validate_split_outputs", validate)
    monkeypatch.setattr(preprocess, "_process_cohort", process)
    monkeypatch.setattr(preprocess, "setup_logger", lambda *args, **kwargs: Mock())
    report = {}
    preprocess.generate(SimpleNamespace(continue_on_error=False), manifest, DicomAudit(), report)
    assert report["completion"]["success"]
    cohort.verify_unchanged()


def test_main_persists_first_error_without_starting_generation(tmp_path, monkeypatch):
    import pytest
    import preprocess

    splits = write_splits(tmp_path / "splits", {})
    called = []
    def fail_during_inventory(args, audit, cohort):
        audit.add_series(path="source/extra", patient_id=None, patient_id_raw="extra",
                         expected_patient=False, series_instance_uid="series-extra",
                         reason="outside_all_splits", file_paths=[])
        raise ValueError("unexpected patient")
    monkeypatch.setattr(preprocess, "prepare", fail_during_inventory)
    monkeypatch.setattr(preprocess, "generate", lambda *args: called.append(True))
    with pytest.raises(SystemExit) as error:
        preprocess.main(["--splits-dir", str(splits.splits_dir), "--output-root", str(tmp_path / "out")])
    assert error.value.code == 1
    assert not called
    audit_dir = tmp_path / "out" / "audit"
    exams = json.loads((audit_dir / "exams.json").read_text(encoding="utf-8"))
    report = json.loads((audit_dir / "run.json").read_text(encoding="utf-8"))
    assert exams[0]["patient_id_raw"] == "extra"
    assert exams[0]["should_include"] is False
    assert exams[0]["final_status"] == "excluded"
    assert report["status"] == "failed"
    assert report["fatal_error"]["message"] == "unexpected patient"
    assert report["patient_coverage"]["unexpected_ids"] == ["extra"]


def test_interrupted_inventory_does_not_claim_unverified_patients_are_missing(tmp_path):
    import preprocess

    audit = DicomAudit()
    report = {"status": "failed", "inventory_complete": False,
              "cohort_snapshot": {"memberships": [{"patient_id": "not-seen"}]}}
    preprocess._persist_audits(audit, report, tmp_path)
    coverage = json.loads((tmp_path / "run.json").read_text())["patient_coverage"]
    assert coverage["missing_ids"] == []
    assert coverage["not_yet_verified_ids"] == ["not-seen"]


def test_output_inside_raw_input_is_rejected_before_writing(tmp_path):
    import pytest
    import preprocess

    raw = tmp_path / "raw"
    with pytest.raises(SystemExit) as error:
        preprocess.main(["--data-root", str(raw), "--output-root", str(raw / "audit")])
    assert error.value.code == 2
    assert not raw.exists()


def test_non_selected_series_never_inherits_patient_processing_success():
    import preprocess

    audit = DicomAudit()
    audit.add_file(path="chosen", patient_id="a", selection_status="selected")
    chosen = audit.add_series(patient_id="a", file_paths=["chosen"], eligibility_status="eligible")
    rejected = audit.add_series(patient_id="a", file_paths=["other"], eligibility_status="eligible")
    preprocess._record_series_results(audit, {"a": {"synthetic_train": {"status": "processed"}}})
    assert chosen["selection_status"] == "selected"
    assert chosen["processing_status"] == "processed"
    assert rejected["selection_status"] == "not_selected"
    assert rejected["processing_status"] == "not_applicable"
    assert preprocess._destination_status({"a": {"status": "reused"}}) == "reused"
    assert preprocess._destination_status({"a": {"status": "processed"},
                                           "b": {"status": "processing_failed"}}) == "partially_processed"


def test_continuation_does_not_process_a_patient_with_blocked_selection(tmp_path, monkeypatch):
    import preprocess

    cohort = write_splits(tmp_path / "splits", {"synthetic_train.txt": ["bad", "good"]})
    called = []
    def process(entry, *args):
        called.append(entry["patient_id"])
        return {"synthetic_train": {"status": "processed"}}
    monkeypatch.setattr(preprocess, "process_patient", process)
    results, failures, events = preprocess._process_cohort(
        cohort, {key: {"patient_id": key} for key in cohort.all_ids}, tmp_path, None, True,
        [{"patient_id": "bad", "reason": "no_valid_ct_candidate"}], lambda *args: None)
    assert called == ["good"]
    assert results["bad"]["synthetic_train"]["status"] == "not_attempted"
    assert failures[0]["reason"] == "no_valid_ct_candidate"


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


def test_prepare_can_select_synthetic_ct_without_any_pixel_or_hdf5_work(tmp_path, monkeypatch):
    import preprocess

    cohort = write_splits(tmp_path / "splits", {"synthetic_train.txt": ["patient"], "mixed_train.txt": ["patient"]})
    root = tmp_path / "data"
    (root / "source").mkdir(parents=True)
    (root / "patient-selection-annotation.json").write_text('{"patient": "selected"}', encoding="utf-8")
    (root / "xrays-view-annotation.json").write_text("{}", encoding="utf-8")
    path = root / "source" / "slice.dcm"
    ct = SimpleNamespace(path_directory=path.parent, _paths_to_dicom_files=[path],
                         acquisition_time={}, is_valid_axial_scan=True,
                         get_array=Mock(side_effect=AssertionError("pixels must not be read")))
    class Dataset:
        def __init__(self, *args, audit, **kwargs):
            self.dataset = {"patient": {"cts": [ct], "xrays": [], "multiview": []}}
            audit.add_file(path=str(path), patient_id="patient", eligibility_status="eligible")
        def load(self):
            pass
        def __getitem__(self, patient):
            return self.dataset[patient]
        write_summary = Mock()
        write_counts = Mock()
    monkeypatch.setattr(preprocess, "DicomDataset", Dataset)
    monkeypatch.setattr(preprocess, "XRayJsonClassifier", lambda **kwargs: SimpleNamespace(
        json_path=root / "xrays-view-annotation.json", load_json=Mock()))
    monkeypatch.setattr(preprocess, "setup_logger", lambda *args, **kwargs: Mock())
    def forbidden(*args, **kwargs):
        raise AssertionError("generation must not run during prepare")
    monkeypatch.setattr("lib.data.generation.PatientDataStorer", forbidden)
    args = SimpleNamespace(data_root=root, output_root=tmp_path / "output", splits_dir=cohort.splits_dir,
                           dataset=["source"], ct_selection=None)
    manifest = preprocess.prepare(args)
    assert manifest["patients"][0]["selected"]["ct"]["paths"] == [str(path)]
    assert manifest["ready_for_generate"]
    assert not list((tmp_path / "output").rglob("*.h5"))
    ct.get_array.assert_not_called()
    cohort.verify_unchanged()
