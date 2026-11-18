"""Read-only cohort contracts for the six versioned experiment splits."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
# Original study reference confirmed by the study owner; never regenerate from outputs.
ORIGINAL_PATIENT_SELECTION_SHA256 = "1e1dfc50594f6486c870c063cdfd0bbc36f45f527cf2efce4abdc859210d5df4"
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


def read_original_annotations(path):
    """Read the authorized immutable reference, never a regenerated replacement."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ORIGINAL_PATIENT_SELECTION_SHA256:
        raise CohortError("Patient selection reference differs from the authorized original; processing blocked")
    return json.loads(raw.decode("utf-8")), digest


def verify_patient_membership(discovered_ids, selected_ids, annotations):
    """Check exact legacy annotation membership without normalizing or changing IDs."""
    discovered, actual_selected = set(discovered_ids), set(selected_ids)
    expected_selected = {patient for patient in discovered if annotations.get(patient) == "selected"}
    expected_not_selected = discovered - expected_selected
    actual_not_selected = discovered - actual_selected
    if actual_selected != expected_selected or actual_not_selected != expected_not_selected:
        raise CohortError("Patient identities or memberships differ from the original selection annotations")
    return {
        "status": "matched_observed_scope", "scope": "patients_discovered_by_legacy_loader",
        "selected_patient_ids": sorted(actual_selected),
        "not_selected_patient_ids": sorted(actual_not_selected),
        "original_reference_ids_not_observed": sorted(set(annotations) - discovered),
        "selected_differences": [], "not_selected_differences": [],
        "full_study_reproduction_verified": False,
    }


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
        self.active_patient_ids: set[str] | None = None
        self.subset_by_dataset: dict[str, set[str]] | None = None
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
    def processing_ids(self) -> tuple[str, ...]:
        if self.active_patient_ids is None:
            return self.all_ids
        return tuple(sorted(self.active_patient_ids))

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

    def check_manifest_integrity(self, manifest: dict) -> tuple[dict, list[dict]]:
        try:
            self.verify_unchanged()
            final_hashes = {name: hashlib.sha256((self.splits_dir / name).read_bytes()).hexdigest()
                            for name in self.splits}
            return {
                "split_hashes_unchanged": True,
                "initial_split_hashes": manifest["split_hashes"],
                "final_split_hashes": final_hashes,
            }, []
        except Exception as exc:
            result = {
                "split_hashes_unchanged": False,
                "initial_split_hashes": manifest.get("split_hashes", {}),
                "split_integrity_error": {"type": type(exc).__name__, "message": str(exc)},
            }
            return result, [{"reason": "protected_split_changed_during_generation", **result}]

    def select_subset(self, data_root: str | Path, datasets: list[str], count: int) -> dict:
        """Apply and describe the deterministic, per-collection patient subset."""
        from lib.data.inventory import SUBSET_ALGORITHM, SUBSET_SEED, select_subset_by_dataset

        if self.subset_by_dataset is None:
            selected, details, index_hashes = select_subset_by_dataset(data_root, datasets, self, count)
            self.subset_by_dataset = selected
            self.subset_index_hashes = index_hashes
            self.subset_details = details
        else:
            selected = self.subset_by_dataset
            details = self.subset_details
            index_hashes = self.subset_index_hashes
        self.active_patient_ids = set().union(*selected.values()) if selected else set()
        return {
            "count_per_dataset": count,
            "seed": SUBSET_SEED,
            "algorithm": SUBSET_ALGORITHM,
            "datasets": details,
            "selected_by_dataset": {name: sorted(ids) for name, ids in selected.items()},
            "active_patient_ids": sorted(self.active_patient_ids),
            "metadata_index_sha256": index_hashes,
        }

    def restore_subset(self, manifest: dict) -> None:
        """Restore the manifest's authorized subset and verify its source indexes."""
        subset = manifest.get("subset")
        if subset is None:
            return
        self.active_patient_ids = set(manifest["active_patient_ids"])
        self.subset_by_dataset = {
            name: set(patient_ids) for name, patient_ids in subset["selected_by_dataset"].items()
        }
        selected_union = set().union(*self.subset_by_dataset.values()) if self.subset_by_dataset else set()
        if selected_union != self.active_patient_ids or not selected_union <= set(self.all_ids):
            raise CohortError("Subset scope differs from selected authorized patients")
        self.subset_index_hashes = manifest.get("subset_index_sha256", {})
        for dataset_name, expected_hash in self.subset_index_hashes.items():
            index_path = Path(manifest["data_root"]) / dataset_name / "metadata.csv"
            if hashlib.sha256(index_path.read_bytes()).hexdigest() != expected_hash:
                raise CohortError(f"Collection index changed after subset selection: {index_path}")

    @staticmethod
    def require_selection(entry: dict, continue_on_error: bool = False) -> None:
        """Enforce the minimum selected exams required by this patient's split."""
        if continue_on_error:
            return
        memberships = entry["memberships"]
        roles = ("ct", "frontal", "lateral") if any(name.startswith("real_") for name in memberships) else ("ct",)
        missing = [role for role in roles if not entry.get("selected", {}).get(role)]
        if missing or entry["annotation_conflict"]:
            raise CohortError(f"Patient {entry['patient_id']}: missing required exams {missing}; "
                              f"annotation_conflict={entry['annotation_conflict']}")

    def expected_ids(self, split_name: str) -> tuple[str, ...]:
        try:
            ids = self.splits[split_name].ids
        except KeyError as exc:
            raise KeyError(f"Unknown split {split_name!r}; expected one of {SPLIT_FILES}") from exc
        if self.active_patient_ids is None:
            return ids
        return tuple(patient_id for patient_id in ids if patient_id in self.active_patient_ids)
