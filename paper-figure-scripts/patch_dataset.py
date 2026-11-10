#!/usr/bin/env python3
"""
CLI to patch ct_xray_data.h5 files by renaming real_xray1/real_xray2 -> xray1/xray2

Usage:
    python patch_dataset.py /path/to/input_dir /path/to/output_dir

This script searches recursively for files named `ct_xray_data.h5` under the
input directory, and for each file that contains both `real_xray1` and
`real_xray2` at the top level it creates a copy in the output directory where
those two datasets are renamed to `xray1` and `xray2`. Other items in the HDF5
file are copied unchanged. Directory structure under the input root is
preserved under the output root.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable
import json
import sys

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data.cohort import Cohort


def find_ct_files(input_dir: Path) -> Iterable[Path]:
    """Yield all ct_xray_data.h5 files under input_dir recursively."""
    return input_dir.rglob("ct_xray_data.h5")


def copy_h5(
    fin: h5py.File, fout: h5py.File, replace_xrays: bool, in_path: Path, out_path: Path
):
    target_names = {"xray1", "xray2"}
    for name in fin:
        if replace_xrays and name == "real_xray1":
            logging.debug("Renaming real_xray1 -> xray1 in %s", out_path)
            fin.copy("real_xray1", fout)
            fin.copy("real_xray1", fout, name="xray1")
        elif replace_xrays and name == "real_xray2":
            logging.debug("Renaming real_xray2 -> xray2 in %s", out_path)
            fin.copy("real_xray2", fout)
            fin.copy("real_xray2", fout, name="xray2")
        else:
            if replace_xrays and name in target_names:
                logging.debug(
                    "Skipping input name '%s' to avoid collision in %s",
                    name,
                    out_path,
                )
                continue
            fin.copy(name, fout)


def _atomic_copy(fin: h5py.File, target: Path, replace_xrays: bool, in_path: Path) -> None:
    temporary = target.with_name(f"{target.stem}.tmp-{os.getpid()}{target.suffix}")
    try:
        with h5py.File(temporary, "w") as fout:
            copy_h5(fin, fout, replace_xrays, in_path, target)
            # Keep source JSON byte-identical; describe the derived product separately.
            fout.attrs["product_destination"] = target.parent.parent.name
            fout.attrs["product_source_path"] = str(in_path)
            fout.attrs["training_xray_source"] = "real_xray" if replace_xrays else "drr"
        with h5py.File(temporary, "r") as check:
            required = {"ct", "xray1", "xray2"}
            if not required.issubset(check.keys()):
                raise ValueError(f"Patched HDF5 misses required keys: {sorted(required - set(check.keys()))}")
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def process_file(
    in_path: Path,
    out_path: Path,
    mixed_out_path: Path,
    *,
    write_pure: bool = True,
    write_mixed: bool = True,
) -> tuple[bool, bool]:
    """Create a patched copy of the HDF5 file at out_path if both
    `real_xray1` and `real_xray2` exist in the input file. Also creates a mixed dataset copy.
    """
    logging.debug("Processing '%s'", in_path)

    processed_pure = False
    processed_mixed = False

    with h5py.File(in_path, "r") as fin:
        has_rx1 = "real_xray1" in fin
        has_rx2 = "real_xray2" in fin
        replace_xrays = has_rx1 and has_rx2

        if write_pure and replace_xrays:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_copy(fin, out_path, True, in_path)
            logging.info("Wrote patched pure file: %s", out_path)
            processed_pure = True
        elif write_pure:
            logging.info("Skipping pure patch for '%s' (missing real_xray1/2)", in_path)

        if write_mixed:
            mixed_out_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_copy(fin, mixed_out_path, replace_xrays, in_path)
            logging.info("Wrote patched mixed file: %s", mixed_out_path)
            processed_mixed = True

    return processed_pure, processed_mixed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch ct_xray_data.h5 files by renaming real_xray1/2 -> xray1/2"
    )

    parser.add_argument("input_dir", type=Path, help="Input directory to search")
    parser.add_argument(
        "output_dir", type=Path, help="Output directory to write patched files"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Read-only directory containing the six authoritative split files",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("Input and output trees must be disjoint; patching never edits source HDF5 files")
    cohort = Cohort(args.splits_dir)
    split_bytes = {name: split.raw_bytes for name, split in cohort.splits.items()}
    def read_membership(names: tuple[str, ...]) -> dict[str, str]:
        ids = {}
        for name in names:
            for patient_id in cohort.expected_ids(name):
                ids[patient_id] = name.removesuffix(".txt")
        return ids

    real_membership = read_membership(("real_train.txt", "real_test.txt"))
    mixed_membership = read_membership(("mixed_train.txt", "mixed_test.txt"))
    real_ids = set(real_membership)
    mixed_ids = set(mixed_membership)

    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 2

    found = list(find_ct_files(input_dir))
    sources_by_patient = {}
    for source in found:
        sources_by_patient.setdefault(source.parent.name, []).append(source)
    duplicates = {key: [str(path) for path in paths] for key, paths in sources_by_patient.items() if len(paths) > 1}
    if duplicates:
        raise ValueError(f"Ambiguous duplicate source HDF5 files: {duplicates}")
    logging.info("Found %d ct_xray_data.h5 files under %s", len(found), input_dir)

    processed_pure_count = 0
    processed_mixed_count = 0
    skipped = 0
    audit_rows = []

    for in_file in found:
        patient_id = in_file.parent.name
        rel = Path(patient_id) / "ct_xray_data.h5"

        real_out_file = output_dir / "real" / real_membership[patient_id] / rel if patient_id in real_membership else output_dir / "real" / "unauthorized" / rel
        mixed_out_file = output_dir / "mixed" / mixed_membership[patient_id] / rel if patient_id in mixed_membership else output_dir / "mixed" / "unauthorized" / rel

        if patient_id not in real_ids and patient_id not in mixed_ids:
            audit_rows.append({"patient_id": patient_id, "source": str(in_file), "status": "outside_all_splits"})
            skipped += 1
            continue

        try:
            p_pure, p_mixed = process_file(
                in_file,
                real_out_file,
                mixed_out_file,
                write_pure=patient_id in real_ids,
                write_mixed=patient_id in mixed_ids,
            )
            if p_pure:
                processed_pure_count += 1
            if p_mixed:
                processed_mixed_count += 1
            audit_rows.append({
                "patient_id": patient_id,
                "source": str(in_file),
                "real_membership": patient_id in real_ids,
                "mixed_membership": patient_id in mixed_ids,
                "pure_written": p_pure,
                "mixed_written": p_mixed,
                "status": "processed" if p_pure or p_mixed else "skipped_missing_required_views",
            })
        except Exception as exc:
            logging.exception("Error processing '%s': %s", in_file, exc)
            skipped += 1
            audit_rows.append({
                "patient_id": patient_id,
                "source": str(in_file),
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            })
            continue

    coverage_failed = False
    for variant, memberships in (("real", real_membership), ("mixed", mixed_membership)):
        for split_name in sorted(set(memberships.values())):
            expected = {patient_id for patient_id, split in memberships.items() if split == split_name}
            root = output_dir / variant / split_name
            by_id = {}
            for path in root.rglob("ct_xray_data.h5") if root.exists() else []:
                by_id.setdefault(path.parent.name, []).append(str(path))
            found_ids = set(by_id)
            duplicates = {key: paths for key, paths in by_id.items() if len(paths) > 1}
            missing = sorted(expected - found_ids)
            unexpected = sorted(found_ids - expected)
            audit_rows.append({
                "split": split_name,
                "expected_ids": sorted(expected),
                "written_ids": sorted(found_ids),
                "missing_ids": missing,
                "unexpected_ids": unexpected,
                "duplicate_outputs": duplicates,
                "status": "validated" if not (missing or unexpected or duplicates) else "coverage_failed",
            })
            coverage_failed = coverage_failed or bool(missing or unexpected or duplicates)

    changed_splits = [
        name for name, original in split_bytes.items()
        if not (args.splits_dir / name).is_file() or (args.splits_dir / name).read_bytes() != original
    ]
    if changed_splits:
        audit_rows.append({"status": "split_integrity_failed", "changed_splits": changed_splits})
        coverage_failed = True

    audit_path = output_dir / "audit" / "patch_events.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps({"schema_version": 1, "events": audit_rows}, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    logging.info(
        "Completed. Pure instances saved: %d, Mixed instances saved: %d, Skipped: %d",
        processed_pure_count,
        processed_mixed_count,
        skipped,
    )
    return 1 if skipped or coverage_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
