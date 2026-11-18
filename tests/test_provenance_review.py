"""Small regressions for audit isolation, publication and metadata-only exports."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lib.data.temporal import observe_dataset, observed_interval


def test_audit_error_cannot_exclude_a_valid_directory(tmp_path, monkeypatch):
    from lib.data.dicom import DicomDataset

    folder = tmp_path / "collection" / "ct"
    folder.mkdir(parents=True)
    (folder / "slice.dcm").touch()
    directory = SimpleNamespace(
        dataset_agnostic_patient_id="original-patient", is_multiview=False,
        is_ct=True, is_valid_scan=True, is_axial=True, is_singleview=False,
    )
    monkeypatch.setattr("lib.data.dicom.DicomDirectory", lambda path: directory)
    dataset = DicomDataset(tmp_path, accepted_datasets=["collection"], logger=Mock())
    monkeypatch.setattr(dataset, "_record_directory_inventory", Mock(side_effect=ValueError("bad optional tag")))
    dataset.load()
    assert list(dataset.keys()) == ["original-patient"]
    assert dataset["original-patient"]["cts"] == [directory]
    assert dataset.audit.errors[0]["stage"] == "metadata_capture"


def test_optional_tag_error_is_metadata_not_image_failure():
    class Tags(dict):
        def get(self, key, default=None):
            if key == "StudyDate":
                raise ValueError("invalid optional value")
            return super().get(key, default)

    record = observe_dataset(Tags(AcquisitionDate="20240101"), "ct.dcm")
    assert record["field_status"]["StudyDate"] == "invalid"
    assert record["temporal"]["raw"] == "20240101"


def test_patient_filter_preserves_annotation_memberships_and_reference_bytes(tmp_path, monkeypatch):
    from lib.data.dicom import DicomDataset

    class View:
        def __len__(self):
            return 1

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    path = tmp_path / "data/patient-selection-annotation.json"
    original = b'{"keep": "selected", "reject": "excluded"}'
    path.write_bytes(original)
    monkeypatch.setattr("lib.data.cohort.ORIGINAL_PATIENT_SELECTION_SHA256", hashlib.sha256(original).hexdigest())
    dataset = DicomDataset(tmp_path / "data", logger=Mock())
    dataset.dataset = {name: {"cts": [], "xrays": [View()], "multiview": []}
                       for name in ("keep", "reject", "unannotated")}
    for name in dataset.dataset:
        dataset.audit.add_patient(patient_id=name)
    dataset.filter_dataset()
    assert list(dataset.keys()) == ["keep"]
    assert dataset.audit.patients_by_id["reject"]["selection_status"] == "excluded"
    assert dataset.audit.patients_by_id["unannotated"]["selection_status"] == "not_selected"
    assert dataset.selection_preservation["selected_patient_ids"] == ["keep"]
    assert dataset.selection_preservation["not_selected_patient_ids"] == ["reject", "unannotated"]
    assert path.read_bytes() == original


def test_reference_check_blocks_same_count_patient_replacement():
    from lib.data.cohort import CohortError, verify_patient_membership

    with pytest.raises(CohortError, match="identities or memberships"):
        verify_patient_membership(["original", "excluded"], ["excluded"],
                                  {"original": "selected", "excluded": "excluded"})


def test_modified_reference_blocks_filter_before_patient_removal(tmp_path, monkeypatch):
    from lib.data.cohort import CohortError
    from lib.data.dicom import DicomDataset

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/patient-selection-annotation.json").write_text('{"patient": "excluded"}')
    dataset = DicomDataset(tmp_path / "data", logger=Mock())
    dataset.dataset = {"patient": {"cts": [], "xrays": [], "multiview": []}}
    with pytest.raises(CohortError, match="authorized original"):
        dataset.filter_dataset()
    assert list(dataset.keys()) == ["patient"]


def test_inventory_does_not_create_a_record_for_each_unread_ct_file(tmp_path):
    from lib.data.dicom import DicomDataset

    paths = [tmp_path / "ct" / f"{index}.dcm" for index in range(100)]
    directory = SimpleNamespace(
        header={"Modality": "CT", "StudyInstanceUID": "study", "SeriesInstanceUID": "series"},
        representative_path=str(paths[50]), _paths_to_dicom_files=paths, dataset="collection",
    )
    dataset = DicomDataset(tmp_path, logger=Mock())
    dataset.dataset = {"patient": {"cts": [directory], "xrays": [], "multiview": []}}
    dataset._record_directory_inventory(directory, paths[0].parent, "collection", paths, "patient")
    assert dataset.audit.files == []
    assert dataset.audit.series[0]["file_paths"] == []
    assert dataset.audit.series[0]["file_count"] == 100
    assert dataset.audit.series[0]["role"] == "ct_series"


def test_directory_metadata_reuses_the_header_without_another_dicom_read(monkeypatch):
    from lib.dicom import GenericDicomDirectory
    from lib.utils.utils import Cache

    representative = SimpleNamespace(dicom_path=Path("/raw/ct.dcm"), header={"StudyDate": "20240101"})
    directory = GenericDicomDirectory.__new__(GenericDicomDirectory)
    directory.cache = Cache()
    directory.get_middle_dcm = lambda: representative
    monkeypatch.setattr("lib.dicom.pydicom.dcmread", Mock(side_effect=AssertionError("unexpected reread")))
    directory._load_dicom()
    assert directory.header is representative.header


def test_ct_header_audit_never_visits_slices_or_frames(monkeypatch):
    from lib.data.temporal import observe_ct_series_header

    class GeneralHeader(dict):
        def get(self, key, default=None):
            if key in {"PerFrameFunctionalGroupsSequence", "SharedFunctionalGroupsSequence",
                       "SOPInstanceUID", "ImagePositionPatient"}:
                raise AssertionError(f"CT instance/frame metadata accessed: {key}")
            if key == "RespiratorySignalSource":
                raise ValueError("malformed optional tag")
            return super().get(key, default)

    monkeypatch.setattr("pydicom.dcmread", Mock(side_effect=AssertionError("unexpected read")))
    record = observe_ct_series_header(GeneralHeader(
        AcquisitionDate="20240101", StartingRespiratoryPhase="INSPIRATION",
        SeriesDescription="inspiration", StudyInstanceUID="study", SeriesInstanceUID="series",
    ), "/raw/ct")
    assert record["record_format"] == "ct_series_header_v1"
    assert record["temporal"]["raw"] == "20240101"
    respiratory = record["respiratory_evidence"]
    assert respiratory["status"] == "explicit_dicom"
    assert respiratory["textual_evidence"][0]["text"] == "inspiration"
    assert {item["location"] for item in respiratory["phase_evidence"]} == {"dataset"}
    assert record["metadata_errors"] == [
        {"field": "RespiratorySignalSource", "message": "malformed optional tag"},
    ]


@pytest.mark.parametrize("slice_count", [2, 100])
def test_ct_candidates_stay_series_units_through_processing_and_exports(tmp_path, monkeypatch, slice_count):
    import csv
    from lib.data.dicom import DicomDataset
    from lib.io import PatientDataStorer

    monkeypatch.setattr("pydicom.dcmread", Mock(side_effect=AssertionError("unexpected read")))
    dataset = DicomDataset(tmp_path, logger=Mock())
    scans = []
    for name in ("chosen", "other"):
        folder = tmp_path / name
        paths = [folder / f"{index}.dcm" for index in range(slice_count)]
        scans.append(SimpleNamespace(
            header={"Modality": "CT", "StudyInstanceUID": "study", "SeriesInstanceUID": name,
                    "AcquisitionDate": "20240101", "StartingRespiratoryPhase": "INSPIRATION"},
            representative_path=str(paths[0]), _paths_to_dicom_files=paths,
            path_directory=folder, dataset="collection", source_root=tmp_path,
            series_summary={"instance_count_observed": slice_count, "instance_count_used": slice_count,
                            "instance_count_discarded": 0},
        ))
    dataset.dataset = {"patient": {"cts": scans, "xrays": [], "multiview": []}}
    for scan in scans:
        dataset._record_directory_inventory(scan, scan.path_directory, "collection",
                                            scan._paths_to_dicom_files, "patient")
    storer = PatientDataStorer.__new__(PatientDataStorer)
    storer.patient_id = "patient"
    sources = storer._build_source_records(scans[0], None, None, None)
    storer._record_audit_output(dataset.audit, sources, [], tmp_path / "product.h5")
    assert dataset.audit.series[1]["processing_status"] == "not_applicable"
    dataset.audit.record_processing_results({"patient": {"real": {"status": "reused"}}})
    chosen, other = dataset.audit.series
    assert chosen["selection_status"] == "selected"
    assert chosen["processing_status"] == "reused"
    assert other["selection_status"] == "not_selected"
    assert other["processing_status"] == "not_applicable"
    assert "sample_id" not in other
    assert dataset.audit.files == []

    report = {"status": "succeeded", "inventory_complete": True}
    dataset.audit.persist_run_report(report, tmp_path)
    with (tmp_path / "exams.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {row["SeriesInstanceUID"] for row in rows} == {"chosen", "other"}
    assert all(row["StartingRespiratoryPhase"] == "INSPIRATION" for row in rows)
    assert all(row["SOPInstanceUID"] == "" for row in rows)
    assert all(row["ImagePositionPatient_status"] == "not_applicable" for row in rows)
    coverage, = report["metadata_coverage"]
    assert coverage["metadata_unit"] == "ct_series"
    assert coverage["records_in_normal_inventory"] == 2
    assert coverage["field_status_counts"]["AcquisitionDate"] == {"observed": 2}
    assert "SOPInstanceUID" not in coverage["field_status_counts"]
    exams = json.loads((tmp_path / "exams.json").read_text())
    assert len(exams) == 2
    assert {row["final_status"] for row in exams} == {"included", "not_selected"}


def test_ct_selection_is_known_even_when_montage_fails(tmp_path):
    from lib.data.audit import DicomAudit
    from lib.io import PatientDataStorer

    audit = DicomAudit()
    chosen = audit.add_series(patient_id="patient", role="ct_series", path=str(tmp_path / "chosen"),
                              eligibility_status="eligible")
    other = audit.add_series(patient_id="patient", role="ct_series", path=str(tmp_path / "other"),
                             eligibility_status="eligible")
    scan = SimpleNamespace(path_directory=tmp_path / "chosen",
                           get_array=Mock(side_effect=ValueError("pixel decode failed")))
    storer = PatientDataStorer.__new__(PatientDataStorer)
    storer.patient_id = "patient"
    storer.audit = audit
    storer.data_dictionary = {"cts": [scan], "xrays": [], "multiview": []}
    with pytest.raises(ValueError, match="pixel decode failed"):
        storer.store_to_file(tmp_path)
    assert chosen["selection_status"] == "selected"
    assert other["selection_status"] == "not_selected"
    audit.record_processing_results({"patient": {"real": {"status": "processing_failed"}}})
    assert chosen["processing_status"] == "processing_failed"
    assert other["processing_status"] == "not_applicable"


def test_failed_publication_preserves_previous_file_and_cleans_temporary(tmp_path, monkeypatch):
    import h5py
    from lib.io import HDF5FileHandler, PatientDataStorer

    output = tmp_path / "ct_xray_data.h5"
    with h5py.File(output, "w") as stored:
        stored.create_dataset("previous_product", data=[1, 2, 3])
    original_bytes = output.read_bytes()
    monkeypatch.setattr(HDF5FileHandler, "store_array", Mock(side_effect=OSError("storage full")))
    storer = PatientDataStorer.__new__(PatientDataStorer)
    with pytest.raises(OSError, match="storage full"):
        storer._publish_product(output, {"ct": None}, {}, ({}, {}), "lzf")
    assert output.read_bytes() == original_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == [output.name]


def test_atomic_publication_preserves_other_existing_hdf5_datasets(tmp_path):
    import h5py
    import numpy as np
    from lib.io import PatientDataStorer

    output = tmp_path / "ct_xray_data.h5"
    with h5py.File(output, "w") as stored:
        stored.create_dataset("legacy_extension", data=np.asarray([7, 8, 9]))
    storer = PatientDataStorer.__new__(PatientDataStorer)
    storer._publish_product(
        output,
        {"ct": np.zeros((2, 2, 2)), "xray1": np.zeros((2, 2)), "xray2": np.zeros((2, 2))},
        {"age": "050"},
        ({"schema_version": 1}, {"schema_version": 1, "artifacts": {}}),
        "lzf",
    )
    with h5py.File(output, "r") as stored:
        assert np.array_equal(stored["legacy_extension"][:], [7, 8, 9])
        assert "metadata" in stored
        assert "provenance_json" in stored


def test_observed_interval_requires_comparability_and_ct_reference():
    from lib.data.temporal import parse_acquisition_time

    ct = {"temporal_scope": "series_assumed_from_representative_header",
          "temporal": parse_acquisition_time({"AcquisitionDate": "20240101"})}
    rx = {"metadata": {"AcquisitionDate": "20240103"}}
    assert observed_interval(ct, rx)["reason"] == "calendar_comparability_not_demonstrated"
    evidence = {"source": "synthetic test reference", "calendar_comparable": True}
    assert observed_interval(ct, rx, evidence)["value"] == 2
    assert "used_instances" not in ct
    assert observed_interval({**ct, "record_format": "unknown"}, rx, evidence)["reason"] == "unsupported_ct_record_format"


def test_native_run_export_uses_only_consolidated_json(tmp_path, monkeypatch):
    import h5py

    script = Path(__file__).resolve().parents[1] / "paper-figure-scripts/export-clinical-audit.py"
    spec = importlib.util.spec_from_file_location("review_exporter", script)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    from lib.data.temporal import parse_acquisition_time

    ct = {"temporal_scope": "series_assumed_from_representative_header",
          "temporal": parse_acquisition_time({"AcquisitionDate": "20240101"})}
    run = {"schema_version": 1, "run_id": "test-run", "files": [], "series": [],
           "patients": [], "samples": [{"sample_id": "collection::patient", "ct": ct,
                                         "frontal": {"metadata": {"AcquisitionDate": "20240103"}},
                                         "lateral": None}]}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run, default=lambda value: value.isoformat()), encoding="utf-8")
    monkeypatch.setattr(h5py, "File", Mock(side_effect=AssertionError("no HDF5 reads allowed")))
    rows = exporter.export_clinical_audit(path, path, path, tmp_path / "export")
    assert rows[0]["intervals"]["frontal_minus_ct"]["status"] == "unavailable"
    assert (tmp_path / "export" / "exams.csv").exists()
    summary = json.loads((tmp_path / "export" / "clinical_summary.json").read_text())
    assert summary["denominator"] == 1
    assert summary["temporal_conclusion"]["status"] == "inconclusive"
