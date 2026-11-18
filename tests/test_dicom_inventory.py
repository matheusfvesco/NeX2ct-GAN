"""Regression coverage for the existing index helpers, not a new dataset API."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lib.data.audit import DicomAudit
from lib.data.cohort import Cohort, SPLIT_FILES
from lib.data.inventory import annotated_view_paths, read_authorized_index, select_subset_by_dataset


class CohortStub:
    memberships = {"included": ("synthetic_train.txt",)}

    def resolve_patient_id(self, raw):
        return (raw, "matched") if raw in self.memberships else (None, "outside_all_splits")


def write_index(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Series UID", "Collection", "Subject ID", "Study UID", "Modality", "Number of Images", "File Location"])
        writer.writerows(rows)


def test_index_groups_series_from_multiple_paths_without_reading_headers(tmp_path, monkeypatch):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "slice.dcm").touch()
    write_index(tmp_path, [("series", "source", "included", "study", "CT", 1, f"./{name}") for name in ("a", "b")])
    monkeypatch.setattr("pydicom.dcmread", lambda *a, **k: pytest.fail("index must not open DICOM headers"))
    audit = DicomAudit()
    series, = read_authorized_index(tmp_path, CohortStub(), {}, audit)
    assert {path.parent.name for path in series["paths"]} == {"a", "b"}
    assert series["metadata"]["number_of_images_indexed"] == 2
    assert len(audit.files) == 2


def test_unexpected_patient_is_audited_then_stops_by_default(tmp_path):
    write_index(tmp_path, [("series", "source", "outside", "study", "CT", 1, "./missing")])
    audit = DicomAudit()
    with pytest.raises(ValueError, match="outside_all_splits"):
        list(read_authorized_index(tmp_path, CohortStub(), {}, audit))
    assert audit.files == []
    assert audit.series[0]["patient_id_raw"] == "outside"
    assert audit.series[0]["reason"] == "outside_all_splits"


def test_subset_sampling_is_deterministic_stratified_and_counts_unique_patients(tmp_path):
    cohort = CohortStub()
    cohort.memberships = {f"p{i}": ("synthetic_train.txt",) for i in range(6)}
    rows = [(f"s-{collection}-{i}", collection, f"p{i}", "study", "CT", 1, "./missing")
            for collection in ("one", "two") for i in range(6)]
    write_index(tmp_path / "source", rows + [rows[0]])
    first = select_subset_by_dataset(tmp_path, ["source"], cohort, 3)
    write_index(tmp_path / "source", list(reversed(rows)))
    second = select_subset_by_dataset(tmp_path, ["source"], cohort, 3)
    assert first[:2] == second[:2]
    assert all(len(ids) == 3 for ids in first[0].values())
    assert all(item["available_authorized_patients"] == 6 for item in first[1])


def test_subset_groups_collections_inside_a_merged_dataset_folder(tmp_path):
    write_index(tmp_path / "merged", [("a", "one", "included", "study", "CT", 1, "./missing"),
                                      ("b", "two", "included", "study", "CT", 1, "./missing")])
    selected, details, hashes = select_subset_by_dataset(tmp_path, ["merged"], CohortStub(), 3)
    assert selected == {"one": {"included"}, "two": {"included"}}
    assert {row["dataset"] for row in details} == {"one", "two"}
    assert set(hashes) == {"merged"}


def test_subset_is_identical_in_fresh_processes_with_different_hash_seeds(tmp_path):
    write_index(tmp_path / "source", [(f"s{i}", "source", f"p{i}", "study", "CT", 1, "./missing") for i in range(6)])
    script = (
        "import json,sys; from types import SimpleNamespace; "
        "from lib.data.inventory import select_subset_by_dataset; "
        "c=SimpleNamespace(resolve_patient_id=lambda raw:(raw,'matched')); "
        "selected,details,hashes=select_subset_by_dataset(sys.argv[1],['source'],c,3); "
        "print(json.dumps({k:sorted(v) for k,v in selected.items()},sort_keys=True))"
    )
    outputs = [subprocess.check_output([sys.executable, "-c", script, str(tmp_path)],
                                      env={**os.environ, "PYTHONHASHSEED": seed}, text=True)
               for seed in ("1", "99")]
    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])["source"]) == 3


def test_subset_limits_inventory_before_enumerating_unselected_patient_paths(tmp_path):
    (tmp_path / "chosen").mkdir()
    (tmp_path / "chosen" / "slice.dcm").touch()
    write_index(tmp_path, [("a", "source", "included", "study", "CT", 1, "./chosen"),
                           ("b", "source", "other", "study", "CT", 1, "./does-not-exist")])
    cohort = CohortStub()
    cohort.memberships = {**cohort.memberships, "other": ("synthetic_train.txt",)}
    cohort.subset_by_dataset = {"source": {"included"}}
    audit = DicomAudit()
    series, = read_authorized_index(tmp_path, cohort, {}, audit)
    assert series["metadata"]["patient_id"] == "included"
    assert len(audit.files) == 1
    assert audit.series[0]["reason"] == "not_selected_by_subset"


def test_subset_does_not_abort_on_external_index_rows_but_full_run_does(tmp_path):
    write_index(tmp_path, [("s", "source", "outside", "study", "CT", 1, "./missing")])
    cohort = CohortStub()
    cohort.subset_by_dataset = {"source": {"included"}}
    assert list(read_authorized_index(tmp_path, cohort, {}, DicomAudit())) == []
    cohort.subset_by_dataset = None
    with pytest.raises(ValueError, match="outside_all_splits"):
        list(read_authorized_index(tmp_path, cohort, {}, DicomAudit()))


def test_subset_preserves_full_memberships_but_scopes_expected_ids(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in SPLIT_FILES:
        (splits / name).write_text("a\nb\n" if name == "synthetic_train.txt" else "")
    cohort = Cohort(splits)
    original = {name: (splits / name).read_bytes() for name in SPLIT_FILES}
    write_index(tmp_path / "source", [(key, "source", key, "study", "CT", 1, "./missing") for key in ("a", "b")])
    subset = cohort.select_subset(tmp_path, ["source"], 1)
    assert set(cohort.memberships) == {"a", "b"}
    assert len(cohort.expected_ids("synthetic_train.txt")) == 1
    assert set(cohort.expected_ids("synthetic_train.txt")) == set(subset["active_patient_ids"])
    assert {name: (splits / name).read_bytes() for name in SPLIT_FILES} == original


def test_annotation_excludes_series_before_enumerating_its_dicom_paths(tmp_path):
    write_index(tmp_path, [("s", "source", "included", "study", "CT", 1, "./missing")])
    cohort = CohortStub()
    cohort.continue_on_error = True
    audit = DicomAudit()
    assert list(read_authorized_index(tmp_path, cohort, {"included": "excluded"}, audit)) == []
    assert audit.files == []
    assert audit.series[0]["reason"] == "patient_selection_annotation_conflict"


def test_subset_output_validation_allows_prior_authorized_patients_outside_sample(tmp_path, monkeypatch):
    from lib.data.validation import validate_split_outputs

    splits = tmp_path / "splits"
    splits.mkdir()
    for name in SPLIT_FILES:
        (splits / name).write_text("a\nb\n" if name == "synthetic_train.txt" else "")
    cohort = Cohort(splits)
    cohort.active_patient_ids = {"a"}
    roots = {name: tmp_path / name.removesuffix(".txt") for name in SPLIT_FILES}
    for patient in ("a", "b"):
        folder = roots["synthetic_train.txt"] / patient
        folder.mkdir(parents=True)
        (folder / "ct_xray_data.h5").touch()
    verified = []
    monkeypatch.setattr("lib.data.validation._validate_output", lambda path, patient, split: verified.append(patient))
    report = validate_split_outputs(cohort, roots)
    row = next(row for row in report["splits"] if row["split"] == "synthetic_train.txt")
    assert report["success"]
    assert row["out_of_scope_ids"] == ["b"]
    assert row["unexpected_ids"] == []
    assert verified == ["a"]


def test_explicit_negative_view_annotation_is_not_a_header_fallback():
    negative, frontal, missing = map(Path, ("negative.dcm", "frontal.dcm", "missing.dcm"))
    from types import SimpleNamespace
    classifier = SimpleNamespace(classify={negative: "excluded", frontal: "frontal"}.__getitem__)
    assert annotated_view_paths([negative], classifier) == []
    assert annotated_view_paths([negative, frontal, missing], classifier) == [frontal, missing]


def test_complete_path_annotations_do_not_trigger_unlabelled_fallback():
    from types import SimpleNamespace
    frontal, lateral, unknown = map(Path, ("f.dcm", "l.dcm", "unknown.dcm"))
    classifier = SimpleNamespace(classify={frontal: "frontal", lateral: "lateral"}.__getitem__)
    assert annotated_view_paths([unknown, frontal, lateral], classifier) == [frontal, lateral]


def test_missing_uid_stays_null_instead_of_becoming_a_directory_path(tmp_path):
    (tmp_path / "series").mkdir()
    (tmp_path / "series" / "slice.dcm").touch()
    write_index(tmp_path, [("", "source", "included", "study", "CT", "", "./series")])
    audit = DicomAudit()
    series, = read_authorized_index(tmp_path, CohortStub(), {}, audit)
    assert series["metadata"]["series_instance_uid"] is None
    assert series["metadata"]["number_of_images_indexed"] is None
    assert audit.files[0]["series_instance_uid"] is None


@pytest.mark.parametrize(("series", "collection", "message"), [
    ("b", "source", "Conflicting collection index"), ("a", "other", "Conflicting Collection"),
])
def test_conflicting_csv_directory_ownership_is_blocked_before_enumeration(tmp_path, series, collection, message):
    (tmp_path / "series").mkdir()
    write_index(tmp_path, [("a", "source", "included", "study", "CT", 1, "./series"),
                           (series, collection, "included", "study", "CT", 1, "./series")])
    audit = DicomAudit()
    with pytest.raises(ValueError, match=message):
        list(read_authorized_index(tmp_path, CohortStub(), {}, audit))
    assert audit.files == []


def test_missing_csv_columns_fail_explicitly_without_directory_scanning(tmp_path):
    (tmp_path / "metadata.csv").write_text("wrong,columns\nx,y\n")
    with pytest.raises(ValueError, match="lacks columns"):
        list(read_authorized_index(tmp_path, CohortStub(), {}, DicomAudit()))


@pytest.mark.parametrize("collection", ["", "   "])
def test_empty_collection_fails_in_inventory_and_subset(tmp_path, collection):
    write_index(tmp_path, [("series", collection, "included", "study", "CT", 1, "./missing")])
    with pytest.raises(ValueError, match="Collection is missing"):
        list(read_authorized_index(tmp_path, CohortStub(), {}, DicomAudit()))
    with pytest.raises(ValueError, match="Collection is missing"):
        select_subset_by_dataset(tmp_path.parent, [tmp_path.name], CohortStub(), 1)
