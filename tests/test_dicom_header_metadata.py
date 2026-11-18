from lib.dicom import CTSlice, GenericDicomFile
from lib.utils.utils import Cache


def test_cached_header_metadata_access_does_not_reload_pixeldata(tmp_path, monkeypatch):
    header = {"PatientID": "included", "SeriesInstanceUID": "series", "Modality": "CT"}
    calls = []

    def read(path, *, stop_before_pixels=False):
        assert stop_before_pixels, "metadata access must not read PixelData"
        calls.append(path)
        return header

    monkeypatch.setattr("lib.dicom.pydicom.dcmread", read)
    dicom_file = GenericDicomFile.__new__(GenericDicomFile)
    dicom_file.dicom_path, dicom_file.cache = tmp_path / "image.dcm", Cache()
    dicom_file._load_dicom(stop_before_pixels=True)
    dicom_file._unload_dicom()

    assert dicom_file.header is header
    assert dicom_file.get_modality() == "CT"
    assert dicom_file.get_windowing_information() == (None, None)
    assert CTSlice._get_raw_windowing_information(dicom_file) == (None, None, None, None)
    assert calls == [dicom_file.dicom_path]


def test_full_dataset_access_still_reads_pixels_after_cache_is_cleared(tmp_path, monkeypatch):
    from unittest.mock import Mock

    dicom_file = GenericDicomFile.__new__(GenericDicomFile)
    dicom_file.dicom_path, dicom_file.cache = tmp_path / "image.dcm", Cache()
    dicom_file.header = {"Modality": "CT"}
    complete = object()
    reader = Mock(return_value=complete)
    monkeypatch.setattr("lib.dicom.pydicom.dcmread", reader)
    assert dicom_file._get_dicom() is complete
    reader.assert_called_once_with(dicom_file.dicom_path)
