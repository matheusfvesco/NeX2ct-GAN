import json
from types import SimpleNamespace
from unittest.mock import Mock

from lib.data.audit import EventJournal, update_processing_results, write_json
from lib.data.generation import process_patient, load_selected_exams


class CohortStub:
    memberships = {"patient": ("real_train.txt", "synthetic_train.txt")}

    def destinations(self, patient_id):
        return ("real_train", "synthetic_train")

    def resolve_patient_id(self, raw):
        return ("patient", "matched")


def test_failure_is_local_to_destination_and_cache_is_cleared(tmp_path, monkeypatch):
    cache = Mock()
    exams = {"ct": SimpleNamespace(cache=cache), "frontal": None, "lateral": None}
    monkeypatch.setattr("lib.data.generation.is_reusable_output", lambda *args: False)
    monkeypatch.setattr("lib.data.generation.load_selected_exams", lambda *args: exams)
    calls = []
    def write(entry, selected, destination, root, logger):
        calls.append(destination)
        if destination == "real_train":
            raise OSError("destination full")
    monkeypatch.setattr("lib.data.generation._write_destination", write)
    journal = EventJournal(tmp_path / "events.jsonl")
    result = process_patient({"patient_id": "patient"}, CohortStub(), tmp_path, journal)
    assert calls == ["real_train", "synthetic_train"]
    assert result["real_train"]["status"] == "processing_failed"
    assert result["synthetic_train"]["status"] == "processed"
    cache.clear.assert_called_once()
    events = journal.consolidate(tmp_path / "events.json")
    assert {(row["destination"], row["status"]) for row in events if row["status"] != "started"} == {
        ("real_train", "processing_failed"), ("synthetic_train", "processed"),
    }


def test_resume_preserves_valid_destination_and_increments_attempts(tmp_path, monkeypatch):
    entry = {"patient_id": "patient"}
    monkeypatch.setattr("lib.data.generation.is_reusable_output", lambda path, patient, members, destination, entry: destination == "real_train")
    monkeypatch.setattr("lib.data.generation.load_selected_exams", lambda *args: {"ct": SimpleNamespace(cache=Mock()), "frontal": None, "lateral": None})
    writer = Mock()
    monkeypatch.setattr("lib.data.generation._write_destination", writer)
    journal_path = tmp_path / "events.jsonl"
    process_patient(entry, CohortStub(), tmp_path, EventJournal(journal_path))
    result = process_patient(entry, CohortStub(), tmp_path, EventJournal(journal_path))
    assert result["real_train"]["status"] == "reused"
    assert result["real_train"]["attempt"] == 2
    assert result["synthetic_train"]["attempt"] == 2
    assert all(call.args[2] == "synthetic_train" for call in writer.call_args_list)


def test_consolidated_audit_keeps_successful_destination_on_partial_failure(tmp_path):
    path = tmp_path / "audit.json"
    write_json(path, {"schema_version": 1,
                     "files": [{"path": "ct", "patient_id": "patient", "selection_status": "selected"}],
                     "series": [{"patient_id": "patient", "file_paths": ["ct"]}],
                     "patients": [{"patient_id": "patient"}]})
    results = {"patient": {"real_train": {"status": "processing_failed"}, "synthetic_train": {"status": "processed"}}}
    update_processing_results(path, results, CohortStub())
    payload = json.loads(path.read_text())
    patient = payload["patients"][0]
    assert patient["processing_status"] == "partially_processed"
    assert patient["destination_status"] == {"real_train": "processing_failed", "synthetic_train": "processed"}
    assert payload["files"][0]["destination_results"]["synthetic_train"]["status"] == "processed"


def test_loading_ct_uses_only_frozen_instances_even_with_other_neighbor_files(tmp_path, monkeypatch):
    frozen = tmp_path / "selected.dcm"
    record = {"path": str(frozen), "study_instance_uid": "study", "series_instance_uid": "series", "sop_instance_uid": "instance"}
    entry = {"patient_id": "patient", "selected": {"ct": {"path": str(tmp_path), "paths": [str(frozen)], "instances": [record]}}}
    def read(path, *, stop_before_pixels):
        assert str(path) == str(frozen) and stop_before_pixels
        return {"PatientID": "patient", "StudyInstanceUID": "study", "SeriesInstanceUID": "series", "SOPInstanceUID": "instance"}
    monkeypatch.setattr("lib.data.generation.pydicom.dcmread", read)
    constructor = Mock(return_value=object())
    monkeypatch.setattr("lib.data.generation.GenericDicomDirectory", constructor)
    monkeypatch.setattr("lib.data.generation.CTScanDirectory", lambda value: SimpleNamespace(is_valid_axial_scan=True))
    load_selected_exams(entry, CohortStub())
    assert constructor.call_args.kwargs["dicom_paths"] == [str(frozen)]
