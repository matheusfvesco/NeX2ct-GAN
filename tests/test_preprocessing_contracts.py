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
    from preprocess import _manifest_candidate, _selection_fingerprint

    candidate = _manifest_candidate({"acquisition_datetime": datetime(2024, 1, 1, 12), "path": "ct", "_object": object()})
    entry = {"policy": "temporal", "selected": {"ct": candidate}, "memberships": ["real_train.txt"]}
    assert _selection_fingerprint(entry) == _selection_fingerprint(json.loads(json.dumps(entry)))


def test_hdf5_optional_metadata_uses_json_without_breaking_legacy_group(tmp_path):
    import numpy as np
    from lib.io import HDF5FileHandler, PatientDataStorer

    storer = PatientDataStorer.__new__(PatientDataStorer)
    storer.patient_id = "patient-a"
    storer.selection_plan = {
        "destination": "synthetic_train", "memberships": ["synthetic_train.txt"],
        "policy": "largest_valid", "selected": {},
    }
    target = tmp_path / "ct_xray_data.h5"
    temporary = tmp_path / "partial.h5"
    arrays = {"ct": np.zeros((2, 2, 2)), "xray1": np.zeros((2, 2)), "xray2": np.zeros((2, 2))}
    metadata = {"schema_version": 1, "training_xray_source": "drr", "optional": None,
                "evidence": [{"frame": 1, "phase": None}]}
    storer._publish_arrays(target, temporary, "lzf", arrays, {"age": "000"}, metadata)
    with HDF5FileHandler(target, mode="read_only") as stored:
        assert stored.read_json("metadata_json")["optional"] is None
        assert stored.read_dict("metadata") == {"age": "000"}
    assert not temporary.exists()


def test_failed_hdf5_publication_leaves_existing_final_untouched(tmp_path, monkeypatch):
    import numpy as np
    from lib.io import HDF5FileHandler, PatientDataStorer

    storer = PatientDataStorer.__new__(PatientDataStorer)
    storer.patient_id = "patient"
    storer.selection_plan = {"destination": "synthetic_train"}
    target, temporary = tmp_path / "final.h5", tmp_path / "partial.h5"
    target.write_bytes(b"previous final product")
    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")
    monkeypatch.setattr(HDF5FileHandler, "store_array", fail)
    with pytest.raises(OSError, match="simulated storage failure"):
        storer._publish_arrays(target, temporary, "lzf", {"ct": np.zeros((2, 2, 2))}, {}, {"training_xray_source": "drr"})
    assert target.read_bytes() == b"previous final product"
    assert not temporary.exists()


def test_split_path_resolution_preserves_order_and_detects_missing_extra_duplicate(tmp_path):
    from lib.datasets import _resolve_split_paths

    a, b = tmp_path / "a" / "ct_xray_data.h5", tmp_path / "b" / "ct_xray_data.h5"
    assert _resolve_split_paths([b, a], ["a", "b"]) == [a, b]
    for paths in ([a], [a, b, tmp_path / "outside" / "ct_xray_data.h5"], [a, b, tmp_path / "duplicate" / "a" / "ct_xray_data.h5"]):
        with pytest.raises(ValueError, match="coverage mismatch"):
            _resolve_split_paths(paths, ["a", "b"])


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
