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


def test_destination_writes_reuse_drrs_for_the_same_frozen_ct(tmp_path, monkeypatch):
    import numpy as np
    from lib.io import PatientDataStorer
    from lib.utils.utils import Cache

    class Scan:
        def __init__(self):
            self.cache, self.drr_calls = Cache(), 0
            self.header, self.spacing = {}, (1, 1, 1)
        def get_array(self):
            return np.zeros((2, 2, 2))
        @property
        def drr_generator(self):
            self.drr_calls += 1
            return SimpleNamespace(frontal_view=np.ones((2, 2)), lateral_view=np.full((2, 2), 2))
    class Processor:
        def __init__(self, array, scan):
            self.array = array.copy()
        def _transpose(self):
            pass
        def _resize(self):
            pass
    published, sources = [], []
    def publish(self, output, temporary, compression, arrays, legacy, metadata):
        published.append(arrays)
        sources.append(metadata["training_xray_source"])
    monkeypatch.setattr("lib.io.CTProcessor", Processor)
    monkeypatch.setattr(PatientDataStorer, "_publish_arrays", publish)
    ct = Scan()
    for destination, real_views in (("synthetic_train", False), ("mixed_train", False),
                                     ("real_train", True), ("mixed_test", True)):
        plan = {"ct": ct, "destination": destination}
        if real_views:
            plan.update(frontal=SimpleNamespace(get_array=lambda: np.full((2, 2), 3), header={}),
                        lateral=SimpleNamespace(get_array=lambda: np.full((2, 2), 4), header={}))
        PatientDataStorer("patient", {"cts": [ct], "xrays": [], "multiview": []},
                          selection_plan=plan).store_to_file(tmp_path / destination)
    assert ct.drr_calls == 1
    assert published[0]["xray1"] is published[1]["xray1"]
    assert np.array_equal(published[0]["ct"], published[1]["ct"])
    assert sources == ["drr", "drr", "real_xray", "real_xray"]
    assert np.all(published[3]["xray1"] == 3)
