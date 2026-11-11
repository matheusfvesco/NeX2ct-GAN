from pathlib import Path
from types import SimpleNamespace
import pytest

from lib.data.audit import DicomAudit
from lib.data.cohort import Cohort, SPLIT_FILES
from lib.data.inventory import annotated_view_paths, read_authorized_index, select_subset_by_dataset


class CohortStub:
    memberships = {"included": ("real_train.txt",)}

    def resolve_patient_id(self, raw):
        return ("included", "matched") if raw == "included" else (None, "outside_all_splits")


def test_index_groups_series_from_multiple_paths_without_reading_headers(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for name in ("one", "two", "excluded"):
        (root / name).mkdir(parents=True)
        (root / name / "slice.dcm").write_bytes(b"fixture placeholder")
    (root / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        "series-a,LIDC-IDRI,included,study,CT,1,./one\n"
        "series-a,LIDC-IDRI,included,study,CT,1,./two\n"
        "series-x,LIDC-IDRI,included,study,CT,1,./excluded\n", encoding="utf-8"
    )
    audit = DicomAudit()
    result = list(read_authorized_index(root, CohortStub(), {"included": "selected"}, audit))
    assert len(result) == 2
    by_uid = {series["metadata"]["series_instance_uid"]: series for series in result}
    assert len(by_uid["series-a"]["paths"]) == 2
    assert len(by_uid["series-x"]["paths"]) == 1
    assert {row["series_instance_uid"] for row in audit.files} == {"series-a", "series-x"}
    assert by_uid["series-a"]["metadata"]["number_of_images_indexed"] == 2
    assert by_uid["series-a"]["metadata"]["dataset"] == "LIDC-IDRI"
    assert all(row["dataset"] == "LIDC-IDRI" for row in audit.files)


def test_unexpected_patient_is_audited_then_stops_by_default(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    index = root / "metadata.csv"
    index.write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        "series-extra,source,external,study,CT,1,./extra\n", encoding="utf-8"
    )
    audit = DicomAudit()
    with pytest.raises(ValueError, match="outside_all_splits"):
        list(read_authorized_index(root, CohortStub(), {}, audit))
    assert audit.series[0]["patient_id_raw"] == "external"
    assert audit.series[0]["series_instance_uid"] == "series-extra"
    assert audit.series[0]["expected_patient"] is False


def test_subset_sampling_is_deterministic_stratified_and_counts_unique_patients(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for name in SPLIT_FILES:
        ids = [f"p{i}" for i in range(6)] if name == "synthetic_train.txt" else []
        (splits_dir / name).write_text("".join(f"{patient_id}\n" for patient_id in ids), encoding="utf-8")
    cohort = Cohort(splits_dir)
    roots = [tmp_path / name for name in ("dataset-a", "dataset-b", "dataset-c")]
    records = {
        "dataset-a": ["p0", "p1", "p2", "p3", "p4"],
        "dataset-b": ["p1", "p2", "p4", "p5"],
        "dataset-c": ["p0", "p2"],
    }
    for root in roots:
        root.mkdir()
        rows = records[root.name]
        (root / "metadata.csv").write_text(
            "Series UID,Collection,Subject ID,Study UID,Modality,File Location\n" +
            "".join(f"series-{index},collection-{root.name},{patient_id},study,CT,./missing-{index}\n"
                    for index, patient_id in enumerate(rows + rows[:1])),
            encoding="utf-8",
        )

    first = select_subset_by_dataset(tmp_path, list(records), cohort, 3)
    for root in roots:
        rows = records[root.name][::-1]
        (root / "metadata.csv").write_text(
            "Series UID,Collection,Subject ID,Study UID,Modality,File Location\n" +
            "".join(f"series-{index},collection-{root.name},{patient_id},study,CT,./missing-{index}\n"
                    for index, patient_id in enumerate(rows + rows[:1])),
            encoding="utf-8",
        )
    second = select_subset_by_dataset(tmp_path, list(reversed(records)), cohort, 3)
    assert first[0] == second[0]
    assert [len(first[0][f"collection-{name}"]) for name in sorted(records)] == [3, 3, 2]
    assert all(len(set(ids)) == len(ids) for ids in first[0].values())
    assert first[1][0]["selected_patient_ids"] == sorted(first[1][0]["selected_patient_ids"])
    assert first[1][2]["selected_count"] == 2


def test_subset_groups_collections_inside_a_merged_dataset_folder(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for name in SPLIT_FILES:
        ids = [f"p{i}" for i in range(6)] if name == "synthetic_train.txt" else []
        (splits_dir / name).write_text("".join(f"{patient_id}\n" for patient_id in ids), encoding="utf-8")
    cohort = Cohort(splits_dir)
    root = tmp_path / "gan_dataset"
    root.mkdir()
    (root / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,File Location\n"
        "s0,LIDC-IDRI,p0,study,CT,./s0\n"
        "s1,LIDC-IDRI,p1,study,CT,./s1\n"
        "s2,LIDC-IDRI,p2,study,CT,./s2\n"
        "s3,4D-Lung,p3,study,CT,./s3\n"
        "s4,4D-Lung,p4,study,CT,./s4\n"
        "s5,4D-Lung,p5,study,CT,./s5\n", encoding="utf-8")

    selected, details, _ = select_subset_by_dataset(tmp_path, ["gan_dataset"], cohort, 2)

    assert set(selected) == {"LIDC-IDRI", "4D-Lung"}
    assert all(len(patient_ids) == 2 for patient_ids in selected.values())
    assert {row["dataset"] for row in details} == set(selected)


def test_subset_is_identical_in_fresh_processes_with_different_hash_seeds(tmp_path):
    import json
    import os
    import subprocess
    import sys

    source = tmp_path / "source"
    source.mkdir()
    (source / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,File Location\n" +
        "".join(f"s{i},source,p{i},study,CT,./missing\n" for i in range(20)), encoding="utf-8")
    script = (
        "import json,sys; from types import SimpleNamespace; "
        "from lib.data.inventory import select_subset_by_dataset; "
        "cohort=SimpleNamespace(resolve_patient_id=lambda raw:(raw,'matched')); "
        "selected,_,_=select_subset_by_dataset(sys.argv[1],['source'],cohort,3); "
        "print(json.dumps({key:sorted(ids) for key,ids in selected.items()},sort_keys=True))"
    )
    outputs = [subprocess.check_output(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONHASHSEED": seed}, text=True)
        for seed in ("1", "7729")]
    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])["source"]) == 3


def test_subset_limits_inventory_before_enumerating_unselected_patient_paths(tmp_path):
    root = tmp_path / "source"
    chosen = root / "chosen"
    unselected = root / "unselected"
    chosen.mkdir(parents=True)
    unselected.mkdir()
    (chosen / "one.dcm").write_bytes(b"fixture")
    (unselected / "two.dcm").write_bytes(b"fixture")
    (root / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        "chosen-series,source,included,study,CT,1,./chosen\n"
        "unselected-series,source,other,study,CT,1,./unselected\n", encoding="utf-8"
    )
    cohort = CohortStub()
    cohort.memberships = {"included": ("synthetic_train.txt",), "other": ("synthetic_train.txt",)}
    cohort.resolve_patient_id = lambda raw: (raw, "matched")
    cohort.subset_by_dataset = {"source": {"included"}}
    cohort.subset_index_hashes = {}
    audit = DicomAudit()
    result = list(read_authorized_index(root, cohort, {}, audit))
    assert len(result) == 1
    assert [row["path"] for row in audit.files] == [str(chosen / "one.dcm")]
    skipped = next(row for row in audit.series if row.get("series_instance_uid") == "unselected-series")
    assert skipped["scope_status"] == "not_selected_by_subset"
    assert skipped["file_paths"] == []


def test_subset_does_not_abort_on_external_index_rows_but_full_run_does(tmp_path):
    root = tmp_path / "source"
    chosen = root / "chosen"
    chosen.mkdir(parents=True)
    (chosen / "one.dcm").write_bytes(b"fixture")
    (root / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,File Location\n"
        "extra,source,outside,study,CT,./nonexistent\n"
        "chosen,source,included,study,CT,./chosen\n", encoding="utf-8")
    cohort = CohortStub()
    cohort.subset_by_dataset = {"source": {"included"}}
    audit = DicomAudit()
    result = list(read_authorized_index(root, cohort, {}, audit))
    assert len(result) == 1
    assert audit.series[0]["reason"] == "not_selected_by_subset"
    assert audit.series[0]["expected_patient"] is False
    assert [row["patient_id"] for row in audit.files] == ["included"]
    with pytest.raises(ValueError, match="outside_all_splits"):
        list(read_authorized_index(root, CohortStub(), {}, DicomAudit()))


def test_subset_preserves_full_memberships_but_scopes_expected_ids(tmp_path):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for name in SPLIT_FILES:
        ids = ["chosen", "not-chosen"] if name == "synthetic_train.txt" else []
        (splits_dir / name).write_text("".join(f"{patient_id}\n" for patient_id in ids), encoding="utf-8")
    cohort = Cohort(splits_dir)
    cohort.active_patient_ids = {"chosen"}
    assert cohort.all_ids == ("chosen", "not-chosen")
    assert cohort.processing_ids == ("chosen",)
    assert cohort.expected_ids("synthetic_train.txt") == ("chosen",)
    assert cohort.memberships["not-chosen"] == ("synthetic_train.txt",)


def test_subset_output_validation_allows_prior_authorized_patients_outside_sample(tmp_path, monkeypatch):
    from lib.data.validation import validate_split_outputs

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for name in SPLIT_FILES:
        ids = ["chosen", "other"] if name == "synthetic_train.txt" else []
        (splits_dir / name).write_text("".join(f"{patient_id}\n" for patient_id in ids), encoding="utf-8")
    cohort = Cohort(splits_dir)
    cohort.active_patient_ids = {"chosen"}
    synthetic_train = tmp_path / "output" / "synthetic" / "synthetic_train"
    for patient_id in ("chosen", "other"):
        patient_dir = synthetic_train / patient_id
        patient_dir.mkdir(parents=True)
        (patient_dir / "ct_xray_data.h5").touch()
    monkeypatch.setattr("lib.data.validation._validate_output", lambda *args: None)
    roots = {
        name: tmp_path / "output" / name.removesuffix(".txt").split("_")[0] / name.removesuffix(".txt")
        for name in SPLIT_FILES
    }
    report = validate_split_outputs(cohort, roots)
    train = next(row for row in report["splits"] if row["split"] == "synthetic_train.txt")
    assert report["success"]
    assert train["expected_ids"] == ["chosen"]
    assert train["out_of_scope_ids"] == ["other"]
    assert train["unexpected_ids"] == []

    unauthorized = synthetic_train / "unauthorized"
    unauthorized.mkdir()
    (unauthorized / "ct_xray_data.h5").touch()
    report = validate_split_outputs(cohort, roots)
    train = next(row for row in report["splits"] if row["split"] == "synthetic_train.txt")
    assert not report["success"]
    assert train["unexpected_ids"] == ["unauthorized"]


def test_annotation_excludes_series_before_enumerating_its_dicom_paths(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        "series-x,source,included,study,CT,100000,./not-present\n", encoding="utf-8"
    )
    audit = DicomAudit()
    cohort = CohortStub()
    cohort.continue_on_error = True
    result = list(read_authorized_index(root, cohort, {"included": "excluded"}, audit))
    assert result == []
    assert audit.files == []
    assert audit.series[0]["reason"] == "patient_selection_annotation_conflict"


def test_loader_constructs_only_authorized_frozen_series(tmp_path, monkeypatch):
    from lib.data.dicom import DicomDataset

    folder = tmp_path / "source"
    series_a, series_b, external = (folder / name for name in ("a", "b", "external"))
    for directory in (series_a, series_b, external):
        directory.mkdir(parents=True)
        (directory / "slice.dcm").write_bytes(b"fixture placeholder")
    with (folder / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        stream.write("Series UID,Collection,Subject ID,Study UID,Modality,Number of Images,File Location\n")
        stream.write("series-a,x,included,study,CT,1,./a\n")
        stream.write("series-b,x,included,study,CT,1,./b\n")
        stream.write("series-x,x,outside,study,CT,1,./external\n")
    def forbidden_read(*args, **kwargs):
        raise AssertionError("path/CSV filtering must not read DICOM headers")
    calls = []
    def classify(path, *, dicom_paths):
        assert all(Path(item).parent.name != "external" for item in dicom_paths)
        calls.append(tuple(dicom_paths))
        tags = {"PatientID": "included", "Modality": "CT", "StudyInstanceUID": "study",
                "SeriesInstanceUID": f"series-{Path(path).name}"}
        return SimpleNamespace(is_ct=True, is_valid_axial_scan=True, dataset_agnostic_patient_id="included",
                               modality="CT", header=tags)
    monkeypatch.setattr("lib.dicom.pydicom.dcmread", forbidden_read)
    monkeypatch.setattr("lib.data.dicom.DicomDirectory", classify)
    logger = SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
    cohort = CohortStub()
    cohort.continue_on_error = True
    dataset = DicomDataset(tmp_path, accepted_datasets=["source"], cohort=cohort, logger=logger)
    dataset.patient_selection_annotations = {"included": "selected", "outside": "selected"}
    dataset.load()
    assert len(calls) == 2
    assert len(dataset["included"]["cts"]) == 2
    assert all("external" not in row["path"] for row in dataset.audit.files)
    assert all(row["dataset"] == "x" for row in dataset.audit.files + dataset.audit.series)


def test_explicit_negative_view_annotation_is_not_a_header_fallback():
    negative, frontal, missing = map(Path, ("negative.dcm", "frontal.dcm", "missing.dcm"))
    classifier = SimpleNamespace(classify={negative: "excluded", frontal: "frontal"}.__getitem__)
    assert annotated_view_paths([negative], classifier) == []
    assert annotated_view_paths([negative, frontal, missing], classifier) == [frontal, missing]


def test_complete_path_annotations_do_not_trigger_unlabelled_fallback():
    frontal, lateral, unknown = map(Path, ("f.dcm", "l.dcm", "unknown.dcm"))
    classifier = SimpleNamespace(classify={frontal: "frontal", lateral: "lateral"}.__getitem__)
    assert annotated_view_paths([unknown, frontal, lateral], classifier) == [frontal, lateral]


def test_missing_uid_stays_null_instead_of_becoming_a_directory_path(tmp_path):
    directory = tmp_path / "series"
    directory.mkdir()
    (directory / "slice.dcm").write_bytes(b"fixture")
    (tmp_path / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        ",source,included,study,CT,,./series\n", encoding="utf-8")
    audit = DicomAudit()
    result = list(read_authorized_index(tmp_path, CohortStub(), {}, audit))
    assert result[0]["metadata"]["series_instance_uid"] is None
    assert audit.files[0]["series_instance_uid"] is None
    assert result[0]["metadata"]["number_of_images_indexed"] is None


@pytest.mark.parametrize(("series", "collection", "message"), [
    ("b", "source", "Conflicting collection index"),
    ("a", "other", "Conflicting Collection"),
])
def test_conflicting_csv_directory_ownership_is_blocked_before_enumeration(tmp_path, series, collection, message):
    (tmp_path / "series").mkdir()
    (tmp_path / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,File Location\n"
        "a,source,included,study,CT,./series\n"
        f"{series},{collection},included,study,CT,./series\n", encoding="utf-8")
    audit = DicomAudit()
    with pytest.raises(ValueError, match=message):
        list(read_authorized_index(tmp_path, CohortStub(), {}, audit))
    assert audit.files == []


def test_missing_csv_columns_fail_explicitly_without_directory_scanning(tmp_path):
    (tmp_path / "metadata.csv").write_text("wrong,columns\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks columns"):
        list(read_authorized_index(tmp_path, CohortStub(), {}, DicomAudit()))


@pytest.mark.parametrize("collection", ["", "   "])
def test_empty_collection_fails_in_inventory_and_subset(tmp_path, collection):
    (tmp_path / "metadata.csv").write_text(
        "Series UID,Collection,Subject ID,Study UID,Modality,File Location\n"
        f"series,{collection},included,study,CT,./missing\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Collection is missing at .*metadata.csv:2"):
        list(read_authorized_index(tmp_path, CohortStub(), {}, DicomAudit()))
    with pytest.raises(ValueError, match=r"Collection is missing at .*metadata.csv:2"):
        select_subset_by_dataset(tmp_path.parent, [tmp_path.name], CohortStub(), 1)


def test_cached_header_access_does_not_reload_pixeldata(tmp_path, monkeypatch):
    from lib.dicom import CTSlice, GenericDicomFile
    from lib.utils.utils import Cache

    header = {"PatientID": "included", "SeriesInstanceUID": "series", "Modality": "CT"}
    calls = []
    def read(path, *, stop_before_pixels=False):
        assert stop_before_pixels, "metadata access must not read PixelData"
        calls.append(path)
        return header
    monkeypatch.setattr("lib.dicom.pydicom.dcmread", read)
    file = GenericDicomFile.__new__(GenericDicomFile)
    file.dicom_path, file.cache = tmp_path / "image.dcm", Cache()
    file._load_dicom(stop_before_pixels=True)
    file._unload_dicom()
    assert file.header is header
    assert file.header["SeriesInstanceUID"] == "series"
    assert file.get_modality() == "CT"
    assert file.get_windowing_information() == (None, None)
    assert CTSlice._get_raw_windowing_information(file) == (None, None, None, None)
    assert calls == [file.dicom_path]


def test_xray_json_can_be_reloaded_for_the_requested_data_root(tmp_path):
    from lib.utils.dicom import XRayJsonClassifier

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root, role in ((first, "frontal"), (second, "lateral")):
        (root / "xrays-view-annotation.json").write_text('{"x.dcm": "' + role + '"}', encoding="utf-8")
    holder = SimpleNamespace(json_path=first / "xrays-view-annotation.json")
    XRayJsonClassifier.load_json(holder)
    assert holder.annotations["x.dcm"] == "frontal"
    XRayJsonClassifier.load_json(holder, root_data=second)
    assert holder.annotations["x.dcm"] == "lateral"
    (second / "xrays-view-annotation.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="path-to-view JSON object"):
        XRayJsonClassifier.load_json(holder)


def test_legacy_filter_api_still_uses_patient_selection_json(tmp_path):
    from unittest.mock import Mock
    from lib.data.dicom import DicomDataset

    (tmp_path / "patient-selection-annotation.json").write_text(
        '{"included": "selected", "excluded": "excluded"}', encoding="utf-8")
    class View:
        def __len__(self):
            return 1
    dataset = DicomDataset(tmp_path, logger=Mock())
    dataset.dataset = {patient: {"cts": [], "xrays": [View()], "multiview": []}
                       for patient in ("included", "excluded")}
    dataset.filter_dataset()
    assert list(dataset.keys()) == ["included"]
