"""Read-only cohort contracts for the six versioned experiment splits."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
SPLIT_FILES = (
    "real_train.txt",
    "real_test.txt",
    "synthetic_train.txt",
    "synthetic_test.txt",
    "mixed_train.txt",
    "mixed_test.txt",
)
_KNOWN_DATASET_PREFIXES = ("MIDRC-RICORD-1A-", "MIDRC-RICORD-1C-")


class CohortError(ValueError):
    """Raised when split contents are malformed or changed during a run."""


@dataclass(frozen=True)
class SplitSnapshot:
    name: str
    path: str
    sha256: str
    raw_bytes: bytes
    ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "ids": list(self.ids),
            "line_count": len(self.ids),
        }


def normalize_patient_id(value: object) -> str | None:
    """Normalize only known dataset prefixes; never use fuzzy/substring matching."""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    for prefix in _KNOWN_DATASET_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", normalized):
        return None
    return normalized


class Cohort:
    """Immutable in-memory view of split bytes, IDs, memberships, and destinations."""

    def __init__(self, splits_dir: str | Path = "splits") -> None:
        self.splits_dir = Path(splits_dir)
        self.splits: dict[str, SplitSnapshot] = {}
        self.memberships: dict[str, tuple[str, ...]] = {}
        self._read_only_snapshot()

    def _read_only_snapshot(self) -> None:
        memberships: dict[str, list[str]] = {}
        errors: list[str] = []
        snapshots: dict[str, SplitSnapshot] = {}
        for name in SPLIT_FILES:
            path = self.splits_dir / name
            try:
                raw = path.read_bytes()
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"{name}: cannot read UTF-8 split ({type(exc).__name__}: {exc})")
                continue
            lines = text.splitlines()
            seen: set[str] = set()
            ids: list[str] = []
            for line_number, value in enumerate(lines, start=1):
                if not value or value.strip() != value or not normalize_patient_id(value):
                    errors.append(f"{name}:{line_number}: empty or malformed patient ID {value!r}")
                    continue
                if value in seen:
                    errors.append(f"{name}:{line_number}: duplicate patient ID {value!r}")
                    continue
                seen.add(value)
                ids.append(value)
                memberships.setdefault(value, []).append(name)
            snapshots[name] = SplitSnapshot(
                name=name,
                path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(),
                raw_bytes=raw,
                ids=tuple(ids),
            )

        for variant in ("real", "synthetic", "mixed"):
            train = set(snapshots.get(f"{variant}_train.txt", SplitSnapshot("", "", "", b"", ())).ids)
            test = set(snapshots.get(f"{variant}_test.txt", SplitSnapshot("", "", "", b"", ())).ids)
            overlap = sorted(train & test)
            if overlap:
                errors.append(f"{variant}: IDs occur in both train and test: {overlap}")
        if errors:
            raise CohortError("Invalid split cohort:\n" + "\n".join(errors))
        self.splits = snapshots
        self.memberships = {key: tuple(sorted(value)) for key, value in memberships.items()}
        self._normalized_ids: dict[str, list[str]] = {}
        for patient_id in self.memberships:
            self._normalized_ids.setdefault(normalize_patient_id(patient_id), []).append(patient_id)
        ambiguous = {key: ids for key, ids in self._normalized_ids.items() if len(ids) > 1}
        if ambiguous:
            raise CohortError(f"Split IDs collide after the documented normalization: {ambiguous}")

    @property
    def all_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.memberships))

    @property
    def real_ids(self) -> tuple[str, ...]:
        return tuple(sorted(patient_id for patient_id, splits in self.memberships.items() if any(x.startswith("real_") for x in splits)))

    @property
    def synthetic_ids(self) -> tuple[str, ...]:
        return tuple(sorted(patient_id for patient_id, splits in self.memberships.items() if any(x.startswith("synthetic_") for x in splits)))

    @property
    def mixed_ids(self) -> tuple[str, ...]:
        return tuple(sorted(patient_id for patient_id, splits in self.memberships.items() if any(x.startswith("mixed_") for x in splits)))

    def resolve_patient_id(self, raw_id: object) -> tuple[str | None, str]:
        normalized = normalize_patient_id(raw_id)
        if normalized is None:
            return None, "patient_id_unresolved"
        matches = self._normalized_ids.get(normalized, [])
        if len(matches) == 1:
            return matches[0], "matched"
        if len(matches) > 1:
            return None, "patient_id_unresolved"
        return None, "outside_all_splits"

    def destinations(self, patient_id: str) -> tuple[str, ...]:
        if patient_id not in self.memberships:
            return ()
        return tuple(sorted(name.removesuffix(".txt") for name in self.memberships[patient_id]))

    def snapshot(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "splits": [self.splits[name].to_dict() for name in SPLIT_FILES],
            "memberships": [
                {"patient_id": patient_id, "splits": list(self.memberships[patient_id]), "destinations": list(self.destinations(patient_id))}
                for patient_id in self.all_ids
            ],
        }

    def write_snapshot(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def verify_unchanged(self) -> None:
        changed = []
        for name, initial in self.splits.items():
            try:
                current = (self.splits_dir / name).read_bytes()
            except OSError:
                changed.append(name)
                continue
            if current != initial.raw_bytes or hashlib.sha256(current).hexdigest() != initial.sha256:
                changed.append(name)
        if changed:
            raise CohortError(f"Protected split contents changed during processing: {', '.join(changed)}")

    def expected_ids(self, split_name: str) -> tuple[str, ...]:
        try:
            return self.splits[split_name].ids
        except KeyError as exc:
            raise KeyError(f"Unknown split {split_name!r}; expected one of {SPLIT_FILES}") from exc
