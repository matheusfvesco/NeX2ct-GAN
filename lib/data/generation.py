"""Resolve frozen selections using the existing DICOM processing classes."""

from pathlib import Path

from lib.dicom import CTScanDirectory, DicomFile, GenericDicomDirectory
from lib.io import HDF5FileHandler, PatientDataStorer


PROCESSING_CONTRACT_VERSION = 3


def load_selected_exams(entry, cohort):
    """Check frozen paths and representative headers without a second CT header scan."""
    patient_id = entry["patient_id"]
    selected = entry.get("selected", {})
    for role in ("ct", "frontal", "lateral"):
        candidate = selected.get(role)
        if candidate is None:
            continue
        instances = candidate.get("instances", [])
        paths = candidate.get("paths", [])
        if not paths or paths != [instance["path"] for instance in instances]:
            raise ValueError(f"Invalid frozen instance list for {patient_id}/{role}")
        for instance in instances:
            if not Path(instance["path"]).is_file():
                raise FileNotFoundError(f"Frozen DICOM path no longer exists: {instance['path']}")
    ct_record = selected.get("ct")
    if ct_record is None:
        raise ValueError(f"No frozen CT for {patient_id}")
    directory = GenericDicomDirectory(ct_record["path"], dicom_paths=ct_record["paths"])
    ct = CTScanDirectory(directory)
    representative = ct.header
    resolved, status = cohort.resolve_patient_id(representative.get("PatientID"))
    if status != "matched" or resolved != patient_id:
        raise ValueError(f"Representative CT header no longer belongs to {patient_id}")
    if ct_record.get("series_instance_uid") and str(representative.get("SeriesInstanceUID")) != ct_record["series_instance_uid"]:
        raise ValueError(f"Representative CT SeriesInstanceUID changed for {patient_id}")
    if not ct.is_valid_axial_scan:
        raise ValueError(f"Selected CT no longer meets technical criteria: {patient_id}")
    result = {"ct": ct, "frontal": None, "lateral": None}
    for role in ("frontal", "lateral"):
        candidate = selected.get(role)
        if candidate is not None:
            if len(candidate["paths"]) != 1:
                raise ValueError(f"Expected a single selected {role} instance for {patient_id}")
            view = DicomFile(Path(candidate["paths"][0]))
            if not view.is_valid:
                raise ValueError(f"Selected radiograph is no longer technically valid: {patient_id}/{role}")
            resolved, status = cohort.resolve_patient_id(view.patient_id)
            if status != "matched" or resolved != patient_id:
                raise ValueError(f"Selected radiograph no longer belongs to {patient_id}/{role}")
            expected_series = candidate.get("series_instance_uid")
            if expected_series and str(view.header.get("SeriesInstanceUID")) != expected_series:
                raise ValueError(f"Selected radiograph SeriesInstanceUID changed for {patient_id}/{role}")
            result[role] = view
    return result


def destination_root(output_root, destination):
    return Path(output_root) / destination.rsplit("_", 1)[0] / destination


def is_reusable_output(path, patient_id, membership, destination, entry):
    if not path.is_file():
        return False
    try:
        with HDF5FileHandler(path, mode="read_only") as stored:
            stored.validate_required_keys(destination.rsplit("_", 1)[0])
            metadata = stored.read_json("metadata_json")
            provenance = stored.read_json("provenance_json")
        if not isinstance(metadata, dict) or not isinstance(provenance, dict):
            return False
        if provenance.get("selected") != entry.get("selected", {}):
            return False
        return (
            provenance.get("schema_version") == 1
            and provenance.get("processing_contract_version") == PROCESSING_CONTRACT_VERSION
            and provenance.get("patient_id") == patient_id
            and provenance.get("memberships") == membership
            and provenance.get("destination") == destination
            and provenance.get("selection_policy") == entry.get("policy")
            and provenance.get("selection_fingerprint") == entry.get("selection_fingerprint")
            and metadata.get("schema_version") == 1
            and metadata.get("selection_status") == entry.get("status")
        )
    except (OSError, KeyError, ValueError, TypeError):
        return False


def _failed_result(exc, output_path, attempt, reason):
    return {
        "status": "processing_failed", "output_path": str(output_path), "attempt": attempt,
        "error": {"reason": reason, "exception_type": type(exc).__name__, "message": str(exc)},
    }


def _write_destination(entry, exams, destination, output_root, logger):
    plan = {**exams, "destination": destination, "processing_contract_version": PROCESSING_CONTRACT_VERSION,
            **{key: entry.get(key) for key in ("policy", "status", "intervals", "memberships",
                                              "respiratory_evidence", "selection_fingerprint", "selected")}}
    PatientDataStorer(
        entry["patient_id"], {"cts": [exams["ct"]], "xrays": [], "multiview": []},
        logger=logger, selection_plan=plan,
    ).store_to_file(destination_root(output_root, destination))


def _record_unattempted_destinations(patient_id, pending_items, start, results, journal):
    for destination, (path, attempt) in pending_items[start:]:
        results[destination] = {
            "status": "not_attempted", "output_path": str(path),
            "attempt": attempt, "reason": "previous_destination_failed",
        }
        journal.append({"patient_id": patient_id, "destination": destination, **results[destination]})


def _write_pending_destinations(entry, exams, pending_items, output_root, logger, journal):
    patient_id = entry["patient_id"]
    results = {}
    for index, (destination, (path, attempt)) in enumerate(pending_items):
        try:
            _write_destination(entry, exams, destination, output_root, logger)
            results[destination] = {"status": "processed", "output_path": str(path), "attempt": attempt}
        except Exception as exc:
            results[destination] = _failed_result(exc, path, attempt, "processing_failed")
        journal.append({"patient_id": patient_id, "destination": destination, **results[destination]})
        if results[destination]["status"] == "processing_failed" and not getattr(journal, "continue_on_error", False):
            _record_unattempted_destinations(patient_id, pending_items, index + 1, results, journal)
            return results
    return results


def process_patient(entry, cohort, output_root, journal, logger=None):
    """Track each authorized destination independently and continue after local failure."""
    patient_id = entry["patient_id"]
    memberships = list(cohort.memberships[patient_id])
    results, pending = {}, {}
    for destination in cohort.destinations(patient_id):
        path = destination_root(output_root, destination) / patient_id / "ct_xray_data.h5"
        attempt = journal.start_attempt(patient_id, destination, path)
        if is_reusable_output(path, patient_id, memberships, destination, entry):
            results[destination] = {"status": "reused", "output_path": str(path), "attempt": attempt}
            journal.append({"patient_id": patient_id, "destination": destination, **results[destination]})
        else:
            pending[destination] = (path, attempt)
    if not pending:
        return results
    try:
        exams = load_selected_exams(entry, cohort)
    except Exception as exc:
        for destination, (path, attempt) in pending.items():
            results[destination] = _failed_result(exc, path, attempt, "frozen_selection_invalid")
            journal.append({"patient_id": patient_id, "destination": destination, **results[destination]})
        return results
    pending_items = list(pending.items())
    try:
        results.update(_write_pending_destinations(entry, exams, pending_items, output_root, logger, journal))
    finally:
        for exam in exams.values():
            if exam is not None:
                exam.cache.clear()
    return results
