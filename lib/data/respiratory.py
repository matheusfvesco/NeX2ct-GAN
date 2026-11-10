"""Metadata-only respiratory evidence extraction for selected DICOM instances."""

from __future__ import annotations

import re
import unicodedata
from typing import Any
import pydicom


def _selected_header(candidate, path):
    header = getattr(candidate.get("_object"), "header", None)
    if header is not None and str(getattr(header, "filename", "")) == path:
        return header
    return pydicom.dcmread(path, stop_before_pixels=True)


def audit_selected_exams(selected: dict) -> dict:
    """Inspect only frozen instances, retaining per-file errors and frame evidence."""
    result = {}
    for role in ("ct", "frontal", "lateral"):
        candidate = selected.get(role) or {}
        evidence = []
        instances = candidate.get("instances", [])
        sampled = instances
        if role == "ct" and instances:
            sampled = [instances[len(instances) // 2]]
        for instance in sampled:
            try:
                header = _selected_header(candidate, instance["path"])
                record = extract_respiratory_evidence(header, **instance)
            except Exception as exc:
                record = {"status": "unreadable", "path": instance["path"],
                          "error": {"type": type(exc).__name__, "message": str(exc)}}
            evidence.append(record)
        result[role] = {
            "evidence": evidence,
            "instances_total": len(instances),
            "instances_inspected": len(sampled),
            "sampling": "representative_instance; Enhanced CT frame sequence remains intact" if role == "ct" else "selected_view_instance",
        }
    return result


PHASE_TAGS = {
    "NominalPercentageOfRespiratoryPhase": "(0020,9245)",
    "StartingRespiratoryPhase": "(0020,9247)",
    "EndingRespiratoryPhase": "(0020,9249)",
}
AUXILIARY_TAGS = {
    "StartingRespiratoryAmplitude": "(0020,9246)",
    "EndingRespiratoryAmplitude": "(0020,9248)",
    "RespiratoryMotionCompensationTechnique": "(0018,9170)",
    "RespiratorySignalSource": "(0018,9171)",
    "RespiratoryTriggerType": "(0020,9250)",
    "RespiratoryIntervalTime": "(0020,9254)",
    "NominalRespiratoryTriggerDelayTime": "(0020,9255)",
    "ActualRespiratoryTriggerDelayTime": "(0020,9257)",
}
TEXT_FIELDS = ("SeriesDescription", "ProtocolName", "StudyDescription", "ImageComments")
_TEXT_PATTERNS = {
    "inspiration": re.compile(r"\b(?:inspiration|inspiratory|inspiracao|inspiratoria|inhale|inhalation)\b"),
    "expiration": re.compile(r"\b(?:expiration|expiratory|expiracao|expiratoria|exhale|exhalation)\b"),
    "technique_only": re.compile(r"\b(?:breath[ -]?hold|apneia|gating)\b"),
    "4d_context": re.compile(r"\b(?:4d[ -]?ct|ct[ -]?4d|phaseimage|respiratory)\b"),
}


def _scalar(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _value(container: Any, keyword: str) -> Any:
    if isinstance(container, dict):
        return container.get(keyword)
    return getattr(container, keyword, None)


def _sequence(container: Any, keyword: str) -> list:
    value = _value(container, keyword)
    return list(value) if value is not None else []


def _walk(container: Any, location: str):
    """Yield top-level and nested synchronization sequence items with stable origins."""
    yield location, container
    for sequence_name in ("RespiratorySynchronizationSequence",):
        for index, item in enumerate(_sequence(container, sequence_name)):
            yield from _walk(item, f"{location}.{sequence_name}[{index}]")


def _structured_evidence(ds: Any) -> tuple[list[dict], list[dict]]:
    phase: list[dict] = []
    auxiliary: list[dict] = []
    scopes = [("dataset", ds)]
    for keyword, label in (("SharedFunctionalGroupsSequence", "shared"), ("PerFrameFunctionalGroupsSequence", "frame")):
        for index, group in enumerate(_sequence(ds, keyword)):
            scopes.append((f"{label}[{index}]", group))
    for location, scope in scopes:
        for current_location, item in _walk(scope, location):
            for keyword, tag in PHASE_TAGS.items():
                value = _value(item, keyword)
                if value is not None and str(value).strip():
                    phase.append({"keyword": keyword, "tag": tag, "value": _scalar(value), "location": current_location})
            for keyword, tag in AUXILIARY_TAGS.items():
                value = _value(item, keyword)
                if value is not None and str(value).strip():
                    auxiliary.append({"keyword": keyword, "tag": tag, "value": _scalar(value), "location": current_location})
    return phase, auxiliary


def _fold(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(char))


def _text_evidence(ds: Any) -> list[dict]:
    result = []
    for field in TEXT_FIELDS:
        raw = _value(ds, field)
        if raw is None:
            continue
        text = str(raw)
        folded = _fold(text)
        for label, pattern in _TEXT_PATTERNS.items():
            match = pattern.search(folded)
            if match:
                result.append({"field": field, "text": text, "indicator": label, "matched": match.group(0)})
        # Percent and PhaseImage index remain literal textual evidence; never map to a state.
        if _TEXT_PATTERNS["4d_context"].search(folded):
            percent = re.search(r"\b(?:phase|fase)\s*[_:-]?\s*(\d{1,3})\s*%", folded)
            if percent and int(percent.group(1)) <= 100:
                result.append({"field": field, "text": text, "indicator": "textual_percentage", "value": int(percent.group(1)), "unit": "percent_literal"})
        index = re.search(r"\bphaseimage[_-]?(\d{1,3})\b", folded)
        if index:
            result.append({"field": field, "text": text, "indicator": "textual_phase_index", "value": int(index.group(1)), "unit": "index"})
    return result


def extract_respiratory_evidence(
    ds: Any,
    *,
    path: str | None = None,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    sop_instance_uid: str | None = None,
) -> dict:
    """Extract evidence from an already selected DICOM header/dataset only."""
    try:
        phase, auxiliary = _structured_evidence(ds)
        textual = _text_evidence(ds)
    except Exception as exc:
        return {
            "status": "unreadable",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "path": path,
            "study_instance_uid": study_instance_uid,
            "series_instance_uid": series_instance_uid,
            "sop_instance_uid": sop_instance_uid,
            "phase_evidence": [],
            "auxiliary_evidence": [],
            "textual_evidence": [],
        }
    if phase:
        status = "explicit_dicom"
    elif any(item["indicator"] in {"inspiration", "expiration", "textual_percentage", "textual_phase_index"} for item in textual):
        status = "textual_indication"
    elif auxiliary or any(item["indicator"] in {"technique_only", "4d_context"} for item in textual):
        status = "technique_only"
    else:
        status = "unavailable"
    phase_values = {}
    for item in phase:
        phase_values.setdefault(item["keyword"], set()).add(str(item["value"]))
    text_states = {item["indicator"] for item in textual if item["indicator"] in {"inspiration", "expiration"}}
    explicit_states = {str(item["value"]).casefold() for item in phase
                       if item["keyword"] in {"StartingRespiratoryPhase", "EndingRespiratoryPhase"}
                       and str(item["value"]).casefold() in {"inspiration", "expiration"}}
    return {
        "status": status,
        "multiple_phase_values": any(len(values) > 1 for values in phase_values.values()),
        "conflicting_textual_states": len(text_states) > 1,
        "conflicting_sources": len(explicit_states) == 1 and len(text_states) == 1 and explicit_states != text_states,
        "phase_evidence": phase,
        "auxiliary_evidence": auxiliary,
        "textual_evidence": textual,
        "path": path,
        "study_instance_uid": study_instance_uid,
        "series_instance_uid": series_instance_uid,
        "sop_instance_uid": sop_instance_uid,
    }
