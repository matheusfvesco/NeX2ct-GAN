import json
from types import SimpleNamespace
from unittest.mock import Mock

from lib.data.audit import update_processing_results, write_json
from lib.data.generation import load_selected_exams


class CohortStub:
    def destinations(self, patient_id):
        return ("real_train", "synthetic_train")

    def resolve_patient_id(self, raw):
        return ("patient", "matched")


def test_consolidated_audit_keeps_successful_destination_on_partial_failure(tmp_path):
    path = tmp_path / "audit.json"
    write_json(path, {
        "schema_version": 1,
        "files": [{"path": "ct", "patient_id": "patient", "selection_status": "selected"}],
        "series": [{"patient_id": "patient", "file_paths": ["ct"]}],
        "patients": [{"patient_id": "patient"}],
    })
    results = {"patient": {
        "real_train": {"status": "processing_failed"},
        "synthetic_train": {"status": "processed"},
    }}

    update_processing_results(path, results, CohortStub())

    payload = json.loads(path.read_text())
    patient = payload["patients"][0]
    assert patient["processing_status"] == "partially_processed"
    assert patient["destination_status"] == {
        "real_train": "processing_failed", "synthetic_train": "processed",
    }
    assert payload["files"][0]["destination_results"]["synthetic_train"]["status"] == "processed"


def test_loading_ct_uses_only_frozen_instances_even_with_other_neighbor_files(tmp_path, monkeypatch):
    frozen = tmp_path / "selected.dcm"
    frozen.write_bytes(b"fixture placeholder")
    record = {"path": str(frozen), "study_instance_uid": "study", "series_instance_uid": "series", "sop_instance_uid": "instance"}
    entry = {"patient_id": "patient", "selected": {"ct": {"path": str(tmp_path), "paths": [str(frozen)], "instances": [record]}}}

    def forbidden(*args, **kwargs):
        raise AssertionError("No additional per-instance header scan is allowed")

    monkeypatch.setattr("lib.dicom.pydicom.dcmread", forbidden)
    constructor = Mock(return_value=object())
    monkeypatch.setattr("lib.data.generation.GenericDicomDirectory", constructor)
    monkeypatch.setattr("lib.data.generation.CTScanDirectory", lambda value: SimpleNamespace(
        is_valid_axial_scan=True, header={"PatientID": "patient", "SeriesInstanceUID": "series"}))
    load_selected_exams(entry, CohortStub())
    assert constructor.call_args.kwargs["dicom_paths"] == [str(frozen)]
