import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np

from lib.data.audit import write_json
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
    cohort.verify_unchanged()


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
