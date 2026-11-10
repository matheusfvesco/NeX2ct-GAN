from pathlib import Path
from types import SimpleNamespace
import pytest

from lib.data.audit import DicomAudit
from lib.data.inventory import annotated_view_paths, read_authorized_index


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
        "Series UID,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        "series-a,included,study,CT,1,./one\n"
        "series-a,included,study,CT,1,./two\n"
        "series-x,included,study,CT,1,./excluded\n", encoding="utf-8"
    )
    audit = DicomAudit()
    result = list(read_authorized_index(root, CohortStub(), {"included": "selected"}, audit))
    assert len(result) == 2
    by_uid = {series["metadata"]["series_instance_uid"]: series for series in result}
    assert len(by_uid["series-a"]["paths"]) == 2
    assert len(by_uid["series-x"]["paths"]) == 1
    assert {row["series_instance_uid"] for row in audit.files} == {"series-a", "series-x"}
    assert by_uid["series-a"]["metadata"]["number_of_images_indexed"] == 2


def test_annotation_excludes_series_before_enumerating_its_dicom_paths(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "metadata.csv").write_text(
        "Series UID,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        "series-x,included,study,CT,100000,./not-present\n", encoding="utf-8"
    )
    audit = DicomAudit()
    result = list(read_authorized_index(root, CohortStub(), {"included": "excluded"}, audit))
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
    dataset = DicomDataset(tmp_path, accepted_datasets=["source"], cohort=CohortStub(), logger=logger)
    dataset.patient_selection_annotations = {"included": "selected", "outside": "selected"}
    dataset.load()
    assert len(calls) == 2
    assert len(dataset["included"]["cts"]) == 2
    assert all("external" not in row["path"] for row in dataset.audit.files)


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
        "Series UID,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        ",included,study,CT,,./series\n", encoding="utf-8")
    audit = DicomAudit()
    result = list(read_authorized_index(tmp_path, CohortStub(), {}, audit))
    assert result[0]["metadata"]["series_instance_uid"] is None
    assert audit.files[0]["series_instance_uid"] is None
    assert result[0]["metadata"]["number_of_images_indexed"] is None


def test_conflicting_csv_directory_ownership_is_blocked_before_enumeration(tmp_path):
    (tmp_path / "series").mkdir()
    (tmp_path / "metadata.csv").write_text(
        "Series UID,Subject ID,Study UID,Modality,File Location\n"
        "a,included,study,CT,./series\n"
        "b,included,study,CT,./series\n", encoding="utf-8")
    audit = DicomAudit()
    with pytest.raises(ValueError, match="Conflicting collection index"):
        list(read_authorized_index(tmp_path, CohortStub(), {}, audit))
    assert audit.files == []


def test_missing_csv_columns_fail_explicitly_without_directory_scanning(tmp_path):
    (tmp_path / "metadata.csv").write_text("wrong,columns\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks columns"):
        list(read_authorized_index(tmp_path, CohortStub(), {}, DicomAudit()))


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
