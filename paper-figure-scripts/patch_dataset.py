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
from pathlib import Path
from typing import Iterable

import h5py


def find_ct_files(input_dir: Path) -> Iterable[Path]:
    """Yield all ct_xray_data.h5 files under input_dir recursively."""
    return input_dir.rglob("ct_xray_data.h5")


def copy_h5(
    fin: h5py.File, fout: h5py.File, replace_xrays: bool, in_path: Path, out_path: Path
):
    target_names = {"xray1", "xray2"}
    for name in fin:
        try:
            if replace_xrays and name == "real_xray1":
                logging.debug("Renaming real_xray1 -> xray1 in %s", out_path)
                fin.copy("real_xray1", fout, name="xray1")
            elif replace_xrays and name == "real_xray2":
                logging.debug("Renaming real_xray2 -> xray2 in %s", out_path)
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
        except Exception as exc:
            logging.warning(
                "Failed to copy '%s' from %s -> %s: %s", name, in_path, out_path, exc
            )


def process_file(
    in_path: Path, out_path: Path, mixed_out_path: Path
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

        if replace_xrays:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w") as fout:
                copy_h5(fin, fout, True, in_path, out_path)
            logging.info("Wrote patched pure file: %s", out_path)
            processed_pure = True
        else:
            logging.info("Skipping pure patch for '%s' (missing real_xray1/2)", in_path)

        mixed_out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(mixed_out_path, "w") as fout:
            copy_h5(fin, fout, replace_xrays, in_path, mixed_out_path)
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
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    mixed_output_dir: Path = output_dir.with_name(output_dir.name + "_mixed")

    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 2

    found = list(find_ct_files(input_dir))
    logging.info("Found %d ct_xray_data.h5 files under %s", len(found), input_dir)

    processed_pure_count = 0
    processed_mixed_count = 0
    skipped = 0

    for in_file in found:
        # preserve directory structure relative to input_dir
        try:
            rel = in_file.relative_to(input_dir)
        except Exception:
            # fallback: use file name only
            rel = in_file.name

        out_file = output_dir.joinpath(rel)
        mixed_out_file = mixed_output_dir.joinpath(rel)

        try:
            p_pure, p_mixed = process_file(in_file, out_file, mixed_out_file)
            if p_pure:
                processed_pure_count += 1
            if p_mixed:
                processed_mixed_count += 1
        except Exception as exc:
            logging.exception("Error processing '%s': %s", in_file, exc)
            skipped += 1
            # try to remove a possibly partially-written file
            try:
                if out_file.exists():
                    out_file.unlink()
                if mixed_out_file.exists():
                    mixed_out_file.unlink()
            except Exception:
                pass
            continue

    logging.info(
        "Completed. Pure instances saved: %d, Mixed instances saved: %d, Skipped: %d",
        processed_pure_count,
        processed_mixed_count,
        skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
