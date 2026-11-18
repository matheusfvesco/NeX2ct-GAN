from datetime import datetime, timedelta
from types import SimpleNamespace
import json

import pytest

from lib.data.cohort import Cohort, CohortError
from lib.data.clinical import validate_interventions
from lib.data.respiratory import extract_respiratory_evidence
from lib.data.selection import select_exams


SPLITS = (
    "real_train.txt",
    "real_test.txt",
    "synthetic_train.txt",
    "synthetic_test.txt",
    "mixed_train.txt",
    "mixed_test.txt",
)


def _write_splits(root, values=None):
    values = values or {}
    root.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        (root / name).write_text("\n".join(values.get(name, [])) + ("\n" if values.get(name) else ""), encoding="utf-8")


def test_cohort_preserves_split_bytes_and_derives_memberships(tmp_path):
    _write_splits(tmp_path, {"real_train.txt": ["patient-a"], "synthetic_train.txt": ["patient-a", "patient-b"]})
    before = {name: (tmp_path / name).read_bytes() for name in SPLITS}
    cohort = Cohort(tmp_path)

    assert cohort.memberships["patient-a"] == ("real_train.txt", "synthetic_train.txt")
    assert cohort.destinations("patient-a") == ("real_train", "synthetic_train")
    assert cohort.resolve_patient_id("patient-b") == ("patient-b", "matched")
    assert cohort.resolve_patient_id("external-patient") == (None, "outside_all_splits")
    cohort.verify_unchanged()
    assert before == {name: (tmp_path / name).read_bytes() for name in SPLITS}


def test_cohort_rejects_duplicates_without_deduplicating(tmp_path):
    _write_splits(tmp_path, {"real_train.txt": ["patient-a", "patient-a"]})
    with pytest.raises(CohortError, match="duplicate patient ID"):
        Cohort(tmp_path)


def test_cohort_detects_external_byte_change_without_restoring_it(tmp_path):
    _write_splits(tmp_path, {"real_train.txt": ["patient-a"]})
    cohort = Cohort(tmp_path)
    path = tmp_path / "real_train.txt"
    changed = b"patient-a\r\n"
    path.write_bytes(changed)
    with pytest.raises(CohortError, match="changed during processing"):
        cohort.verify_unchanged()
    assert path.read_bytes() == changed


@pytest.mark.parametrize("values", [
    {"real_train.txt": ["patient-a"], "real_test.txt": ["patient-a"]},
    {"real_train.txt": [" patient-a"]},
    {"real_train.txt": ["patient-a", ""]},
    {"real_train.txt": ["MIDRC-RICORD-1A-123"], "synthetic_train.txt": ["123"]},
])
def test_cohort_rejects_malformed_overlapping_and_ambiguous_ids(tmp_path, values):
    _write_splits(tmp_path, values)
    before = {name: (tmp_path / name).read_bytes() for name in SPLITS}
    with pytest.raises(CohortError):
        Cohort(tmp_path)
    assert before == {name: (tmp_path / name).read_bytes() for name in SPLITS}


def test_intervention_input_keeps_unassessed_ids_and_validates_scope():
    records = validate_interventions(
        {
            "schema_version": 1,
            "patients": [{"patient_id": "patient-a", "status": "documented_no_event", "source": "clinical-record", "locator": "page-2", "events": []}],
        },
        {"patient-a", "patient-b"},
    )
    assert records["patient-a"]["status"] == "documented_no_event"
    assert records["patient-b"]["status"] == "not_assessed"

    with pytest.raises(ValueError, match="outside the real split union"):
        validate_interventions(
            {"schema_version": 1, "patients": [{"patient_id": "external", "status": "not_assessed"}]},
            {"patient-a"},
        )


def test_temporal_selection_uses_joint_interval_and_stable_ties():
    base = datetime(2024, 1, 1, 12)
    cts = [{"path": "ct-b", "slice_count": 40, "eligible": True, "acquisition_datetime": base, "time_precision": "second"}]
    frontals = [{"path": "frontal", "eligible": True, "acquisition_datetime": base + timedelta(minutes=4), "time_precision": "second"}]
    laterals = [{"path": "lateral", "eligible": True, "acquisition_datetime": base - timedelta(minutes=3), "time_precision": "second"}]

    result = select_exams("temporal", cts, frontals, laterals)

    assert result["status"] == "selected"
    assert result["intervals"]["frontal_minus_ct"]["value"] == 240
    assert result["intervals"]["lateral_minus_ct"]["value"] == -180
    assert result["intervals"]["lateral_minus_frontal"]["value"] == -420


def test_temporal_selection_marks_fallback_for_incomparable_timestamps():
    result = select_exams(
        "temporal",
        [{"path": "ct", "slice_count": 30, "eligible": True}],
        [{"path": "frontal", "eligible": True}],
        [{"path": "lateral", "eligible": True}],
    )
    assert result["status"] == "temporal_fallback"
    assert result["policy"] == "largest_valid"
    assert result["fallback_reason"]


def test_respiratory_evidence_does_not_infer_phase_from_breath_hold_or_percent():
    phase_scope = SimpleNamespace(
        NominalPercentageOfRespiratoryPhase=30,
        StartingRespiratoryPhase=None,
        EndingRespiratoryPhase=None,
    )
    dataset = SimpleNamespace(
        SharedFunctionalGroupsSequence=[],
        PerFrameFunctionalGroupsSequence=[],
        RespiratoryMotionCompensationTechnique="BREATH_HOLD",
        SeriesDescription="4DCT Phase 30%",
        ProtocolName=None,
        StudyDescription=None,
        ImageComments=None,
    )
    dataset.RespiratorySynchronizationSequence = [phase_scope]

    result = extract_respiratory_evidence(dataset)

    assert result["status"] == "explicit_dicom"
    assert any(item["indicator"] == "textual_percentage" for item in result["textual_evidence"])
    assert not any("inspiration" in str(item).lower() for item in result["textual_evidence"])


def test_manifest_fingerprint_survives_json_round_trip():
    from lib.data.selection import manifest_candidate, selection_fingerprint

    candidate = manifest_candidate({"acquisition_datetime": datetime(2024, 1, 1, 12), "path": "ct", "_object": object()})
    entry = {"policy": "temporal", "selected": {"ct": candidate}, "memberships": ["real_train.txt"]}
    assert selection_fingerprint(entry) == selection_fingerprint(json.loads(json.dumps(entry)))


def test_respiratory_multiframe_preserves_scope_without_calling_multiple_phases_a_conflict():
    ds = {
        "SharedFunctionalGroupsSequence": [{"RespiratorySynchronizationSequence": [{"RespiratoryMotionCompensationTechnique": "GATING"}]}],
        "PerFrameFunctionalGroupsSequence": [
            {"RespiratorySynchronizationSequence": [{"NominalPercentageOfRespiratoryPhase": 0}]},
            {"RespiratorySynchronizationSequence": [{"NominalPercentageOfRespiratoryPhase": 50}]},
        ],
    }
    result = extract_respiratory_evidence(ds, path="selected.dcm", series_instance_uid="series")
    assert result["multiple_phase_values"]
    assert not result["conflicting_sources"]
    assert {row["location"] for row in result["phase_evidence"]} == {
        "frame[0].RespiratorySynchronizationSequence[0]", "frame[1].RespiratorySynchronizationSequence[0]",
    }
    assert result["series_instance_uid"] == "series"


@pytest.mark.parametrize(("text", "status"), [
    ("BREATH HOLD", "technique_only"), ("GATING", "technique_only"),
    ("PhaseImage_07", "textual_indication"), ("arterial phase 30%", "unavailable"),
    ("CHEST INSPIRATION", "textual_indication"), ("CHEST EXPIRATION", "textual_indication"),
    ("", "unavailable"),
])
def test_respiratory_text_does_not_infer_anatomical_state_from_technique_or_contrast(text, status):
    result = extract_respiratory_evidence({"SeriesDescription": text})
    assert result["status"] == status


def test_respiratory_structured_and_textual_disagreement_is_explicit():
    result = extract_respiratory_evidence({"StartingRespiratoryPhase": "EXPIRATION", "SeriesDescription": "INSPIRATION"})
    assert result["conflicting_sources"]


def test_respiratory_audit_reuses_the_selected_representative_header(monkeypatch):
    from lib.data.respiratory import audit_selected_exams

    header = SimpleNamespace(filename="selected.dcm", StartingRespiratoryPhase="INSPIRATION")
    candidate = {"instances": [{"path": "selected.dcm"}], "_object": SimpleNamespace(header=header)}
    def forbidden(*args, **kwargs):
        raise AssertionError("the selected header is already available")
    monkeypatch.setattr("lib.data.respiratory.pydicom.dcmread", forbidden)
    result = audit_selected_exams({"ct": candidate})
    assert result["ct"]["evidence"][0]["status"] == "explicit_dicom"


def test_ct_provenance_uses_pixel_reads_without_changing_z_order(tmp_path, monkeypatch):
    import numpy as np
    from lib.dicom import CTScanDirectory

    paths = [tmp_path / "low.dcm", tmp_path / "high.dcm", tmp_path / "discarded.dcm"]
    class Slice(dict):
        @property
        def pixel_array(self):
            return np.full((2, 2), self["ImagePositionPatient"][2])
    datasets = {
        path: Slice(PatientID="patient", SeriesInstanceUID="series", StudyInstanceUID="study",
                    SOPInstanceUID=path.stem, Rows=rows, Columns=columns, ImagePositionPatient=[0, 0, z],
                    AcquisitionDateTime=date)
        for path, z, rows, columns, date in zip(
            paths, (1, 2, 3), (512, 512, 256), (512, 512, 256),
            ("20240101120000", "20240102120000", "20240103120000"),
        )
    }
    calls = []
    def read(path):
        calls.append(path)
        return datasets[path]
    monkeypatch.setattr("lib.dicom.pydicom.dcmread", read)
    ct = CTScanDirectory.__new__(CTScanDirectory)
    ct._paths_to_dicom_files = paths
    ct._header = {"PatientID": "patient", "SeriesInstanceUID": "series"}
    volume = ct._get_raw_array()
    assert calls == paths
    assert np.all(volume[:, :, 0] == 2) and np.all(volume[:, :, 1] == 1)
    assert ct.series_summary["instance_count_observed"] == 3
    assert ct.series_summary["instance_count_used"] == 2
    assert ct.series_summary["instance_count_discarded"] == 1
    assert ct.series_summary["assembly_rule"] == "image_position_patient_z_descending"
    assert not hasattr(ct, "source_instances")


@pytest.mark.parametrize("complete_multiview", [True, False])
def test_hdf5_keeps_ct_and_clinical_view_sources_and_dates(tmp_path, monkeypatch, complete_multiview):
    import csv
    import h5py
    import numpy as np
    from lib.data.audit import DicomAudit
    from lib.io import PatientDataStorer

    class Processor:
        def __init__(self, array, scan):
            self.array = array.copy()
        def _transpose(self):
            pass
        def _resize(self):
            pass

    monkeypatch.setattr("lib.io.CTProcessor", Processor)
    root = tmp_path / "raw"
    ct_path = root / "patient" / "ct"
    ct_path.mkdir(parents=True)
    frontal_path = root / "patient" / "frontal.dcm"
    lateral_path = root / "patient" / "lateral.dcm"
    header = {"PatientAge": "050Y", "AcquisitionDate": "20240101",
              "StudyInstanceUID": "ct-study", "SeriesInstanceUID": "ct-series",
              "SliceThickness": "1"}
    ct = SimpleNamespace(
        path_directory=ct_path, source_root=root,
        dataset="collection-a", header=header,
        dicom=SimpleNamespace(dicom=SimpleNamespace(get=header.get)),
        spacing=(1, 1, 1),
        series_summary={"instance_count_observed": 3, "instance_count_used": 2,
                        "instance_count_discarded": 1,
                        "discard_reason_counts": {"slice_dimensions_not_512x512": 1}},
        get_array=lambda: np.zeros((2, 2, 2)),
        drr_generator=SimpleNamespace(frontal_view=np.zeros((2, 2)), lateral_view=np.zeros((2, 2))),
    )

    def view(path, role):
        metadata = {"path": str(path), "study_instance_uid": role + "-study",
                    "series_instance_uid": role + "-series", "sop_instance_uid": role + "-sop",
                    "temporal": {"raw": role + "-date", "source": "AcquisitionDate"},
                    "metadata": {"AcquisitionDate": role + "-date"}}
        return SimpleNamespace(
            source_root=root, source_metadata=metadata,
            dicom=SimpleNamespace(dicom=header.get),
            get_array=lambda: np.full((2, 2), {
                "frontal": 1, "lateral": 1, "multi-frontal": 2, "multi-lateral": 3,
            }[role]),
        )

    frontal = view(frontal_path, "frontal")
    lateral = view(lateral_path, "lateral")
    multiview_frontal = view(root / "patient" / "multi-frontal.dcm", "multi-frontal")
    multiview_lateral = view(root / "patient" / "multi-lateral.dcm", "multi-lateral")
    multiview_views = {"frontal": multiview_frontal, "lateral": multiview_lateral}
    data = {
        "cts": [ct],
        "xrays": [SimpleNamespace(view_type="frontal", get_view=lambda: frontal,
                                   dicom=SimpleNamespace(dicom=SimpleNamespace(get=header.get))),
                  SimpleNamespace(view_type="lateral", get_view=lambda: lateral,
                                   dicom=SimpleNamespace(dicom=SimpleNamespace(get=header.get)))],
        "multiview": [SimpleNamespace(
            has_both_views=complete_multiview,
            get_by_view_type=lambda role: multiview_views[role] if complete_multiview or role == "frontal" else None,
        )],
    }
    storer = PatientDataStorer.__new__(PatientDataStorer)
    storer.patient_id = "patient"
    storer.data_dictionary = data
    storer.logger = None
    storer.run_id = "run-test"
    audit = DicomAudit()
    audit.add_patient(patient_id="patient", selection_status="selected")
    audit.add_series(path=str(ct_path), patient_id="patient", collection="collection-a",
                     modality="CT", role="ct_series", file_paths=[], file_count=3,
                     series_summary=ct.series_summary, eligibility_status="eligible",
                     selection_status="not_evaluated")
    file_records = (
        (str(frontal_path), "single-frontal-series"),
        (str(lateral_path), "single-lateral-series"),
        (str(multiview_frontal.source_metadata["path"]), "multiview-frontal-series"),
        (str(multiview_lateral.source_metadata["path"]), "multiview-lateral-series"),
    )
    for path, series_uid in file_records:
        audit.add_series(patient_id="patient", collection="collection-a", series_instance_uid=series_uid,
                         file_paths=[path], eligibility_status="eligible", selection_status="not_evaluated")
        audit.add_file(path=path, patient_id="patient", collection="collection-a",
                       series_instance_uid=series_uid, selection_status="not_evaluated")
    storer.audit = audit
    output = tmp_path / "output"
    storer.store_to_file(output)
    audit.persist_run_report({
        "run_id": "run-test", "status": "succeeded", "inventory_complete": True,
        "selection": {"patients": [{"patient_id": "patient"}]},
    }, tmp_path / "audit")

    with h5py.File(output / "patient" / "ct_xray_data.h5", "r") as stored:
        import json
        metadata = json.loads(stored["metadata_json"].asstr()[()])
        provenance = json.loads(stored["provenance_json"].asstr()[()])
        assert metadata["run_id"] == provenance["run_id"] == "run-test"
        assert metadata["sample_id"] == provenance["sample_id"] == "collection-a::patient"
        assert metadata["ct"]["study_instance_uid"] == "ct-study"
        assert metadata["ct"]["series_instance_uid"] == "ct-series"
        assert metadata["ct"]["temporal_scope"] == "series_assumed_from_representative_header"
        assert metadata["ct"]["series_summary"]["instance_count_used"] == 2
        assert "used_instances" not in metadata["ct"]
        assert "discarded_instances" not in metadata["ct"]
        assert metadata["frontal"]["metadata"]["AcquisitionDate"] == "multi-frontal-date"
        expected_lateral = "multi-lateral-date" if complete_multiview else "lateral-date"
        assert metadata["lateral"]["metadata"]["AcquisitionDate"] == expected_lateral
        assert provenance["selected"]["ct"] == str(ct_path)
        assert provenance["artifacts"]["ct_xray_data.h5"] == "generated"
        assert stored["metadata"]["age"].asstr()[()] == "050"
        assert stored["ct"].shape == (2, 2, 2)
        assert np.all(stored["real_xray1"][:] == 2)
        assert np.all(stored["real_xray2"][:] == (3 if complete_multiview else 1))
    with (tmp_path / "audit" / "exams.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = {row["path"]: row for row in csv.DictReader(stream)}
    assert csv_rows[str(multiview_frontal.source_metadata["path"])]["selection_status"] == "selected"
    assert csv_rows[str(frontal_path)]["selection_status"] == "not_selected"
    lateral_source = multiview_lateral if complete_multiview else lateral
    assert csv_rows[str(lateral_source.source_metadata["path"])]["AcquisitionDate"] == expected_lateral
    ct_row = csv_rows[str(ct_path)]
    assert ct_row["role"] == "ct_series"
    assert ct_row["instance_count_used"] == "2"
    assert str(ct_path / "slice.dcm") not in csv_rows


def test_image_array_wrappers_do_not_add_a_duplicate_dicom_read(monkeypatch):
    from unittest.mock import Mock
    import numpy as np
    from lib.dicom import CTSlice, XRayView

    image = np.zeros((2, 2))
    def forbidden(*args, **kwargs):
        raise AssertionError("get_raw_array is responsible for the pixel read")
    monkeypatch.setattr("lib.dicom.pydicom.dcmread", forbidden)
    processor = SimpleNamespace(process=Mock(return_value=image))
    monkeypatch.setattr("lib.dicom.XRayProcessor", Mock(return_value=processor))
    for cls in (XRayView, CTSlice):
        view = cls.__new__(cls)
        view.get_raw_array = Mock(return_value=image)
        assert view.get_array() is image
        view.get_raw_array.assert_called_once_with()
    processor.process.assert_called_once_with()


def test_xray_pixel_read_captures_its_own_dicom_metadata():
    import numpy as np
    from pathlib import Path
    from lib.dicom import GenericDicomFile
    from lib.utils.utils import Cache

    image = np.ones((2, 2))
    class Dicom(dict):
        @property
        def pixel_array(self):
            return image

    dataset = Dicom(
        StudyInstanceUID="xray-study", SeriesInstanceUID="xray-series",
        SOPInstanceUID="xray-image", AcquisitionDateTime="20240203112233",
        PatientPosition="AP",
    )
    view = GenericDicomFile.__new__(GenericDicomFile)
    view.dicom_path = Path("/raw/patient/xray.dcm")
    view.cache = Cache()
    view.cache.add("dicom", dataset)

    assert view.get_raw_array() is image
    assert view.source_metadata["study_instance_uid"] == "xray-study"
    assert view.source_metadata["series_instance_uid"] == "xray-series"
    assert view.source_metadata["sop_instance_uid"] == "xray-image"
    assert view.source_metadata["temporal"]["raw"] == "20240203112233"
    assert view.source_metadata["metadata"]["PatientPosition"] == "AP"
