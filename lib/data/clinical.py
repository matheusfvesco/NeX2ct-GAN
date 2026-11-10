"""Versioned clinical intervention input contract; absence is never a negative finding."""

from __future__ import annotations


INTERVENTION_SCHEMA_VERSION = 1
INTERVENTION_STATUSES = {
    "not_assessed",
    "documented_event",
    "documented_no_event",
    "not_available_in_source",
    "timing_uncertain",
}


def validate_interventions(payload: dict | None, real_ids: set[str]) -> dict[str, dict]:
    """Validate a supplied intervention registry and fill unassessed real IDs explicitly."""
    if payload is None:
        payload = {"schema_version": INTERVENTION_SCHEMA_VERSION, "patients": []}
    if not isinstance(payload, dict) or payload.get("schema_version") != INTERVENTION_SCHEMA_VERSION:
        raise ValueError(f"Intervention input must use schema_version={INTERVENTION_SCHEMA_VERSION}")
    patients = payload.get("patients", [])
    if isinstance(patients, dict):
        rows = [{"patient_id": patient_id, **record} for patient_id, record in patients.items()]
    elif isinstance(patients, list):
        rows = patients
    else:
        raise ValueError("Intervention patients must be a list or an ID-keyed object")
    records = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("patient_id"):
            raise ValueError("Every intervention row requires patient_id")
        patient_id = str(row["patient_id"])
        if patient_id not in real_ids:
            raise ValueError(f"Intervention patient {patient_id!r} is outside the real split union")
        if patient_id in records:
            raise ValueError(f"Duplicate intervention record for patient {patient_id!r}")
        status = row.get("status", "not_assessed")
        if status not in INTERVENTION_STATUSES:
            raise ValueError(f"Unknown intervention status {status!r} for patient {patient_id!r}")
        events = row.get("events", [])
        if not isinstance(events, list):
            raise ValueError(f"Intervention events must be a list for patient {patient_id!r}")
        if status == "documented_event" and not events:
            raise ValueError(f"documented_event requires at least one event for patient {patient_id!r}")
        if status == "documented_no_event" and not (row.get("source") and row.get("locator")):
            raise ValueError(f"documented_no_event requires source and locator for patient {patient_id!r}")
        for index, event in enumerate(events):
            if not isinstance(event, dict) or not (event.get("source") and event.get("locator")):
                raise ValueError(f"Intervention event {index} requires source and locator for patient {patient_id!r}")
        records[patient_id] = {**row, "patient_id": patient_id, "status": status, "events": events}
    return {
        patient_id: records.get(
            patient_id,
            {"patient_id": patient_id, "status": "not_assessed", "events": []},
        )
        for patient_id in sorted(real_ids)
    }
