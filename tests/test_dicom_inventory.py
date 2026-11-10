from pathlib import Path
from types import SimpleNamespace

from lib.data.audit import DicomAudit
from lib.data.inventory import read_authorized_index


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
    result = read_authorized_index(root, CohortStub(), {"included": "selected"}, audit)
    assert len(result) == 2
    by_uid = {series["metadata"]["series_instance_uid"]: series for series in result}
    assert len(by_uid["series-a"]["paths"]) == 2
    assert len(by_uid["series-x"]["paths"]) == 1
    assert {row["series_instance_uid"] for row in audit.files} == {"series-a"}


def test_annotation_excludes_series_before_enumerating_its_dicom_paths(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "metadata.csv").write_text(
        "Series UID,Subject ID,Study UID,Modality,Number of Images,File Location\n"
        "series-x,included,study,CT,100000,./not-present\n", encoding="utf-8"
    )
    audit = DicomAudit()
    result = read_authorized_index(root, CohortStub(), {"included": "excluded"}, audit)
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
        tags = {"Modality": "CT", "StudyInstanceUID": "study",
                "SeriesInstanceUID": f"series-{Path(path).name}"}
        representative = SimpleNamespace(get=lambda name: tags.get(name))
        return SimpleNamespace(is_ct=True, is_valid_axial_scan=True, dataset_agnostic_patient_id="included",
                               modality="CT", dicom=SimpleNamespace(dicom=representative))
    monkeypatch.setattr("lib.dicom.pydicom.dcmread", forbidden_read)
    monkeypatch.setattr("lib.data.dicom.DicomDirectory", classify)
    logger = SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
    dataset = DicomDataset(tmp_path, accepted_datasets=["source"], cohort=CohortStub(), logger=logger)
    dataset.patient_selection_annotations = {"included": "selected", "outside": "selected"}
    dataset.load()
    assert len(calls) == 2
    assert len(dataset["included"]["cts"]) == 2
    assert all("external" not in row["path"] for row in dataset.audit.files)
