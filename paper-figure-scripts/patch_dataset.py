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


def process_file(in_path: Path, out_path: Path) -> None:
    """Create a patched copy of the HDF5 file at out_path if both
    `real_xray1` and `real_xray2` exist in the input file.

    If both keys are present the output file will contain `xray1` and `xray2`
    datasets instead of `real_xray1` and `real_xray2`. Other top-level groups
    and datasets are copied as-is.
    """
    logging.debug("Processing '%s' -> '%s'", in_path, out_path)

    with h5py.File(in_path, "r") as fin:
        has_rx1 = "real_xray1" in fin
        has_rx2 = "real_xray2" in fin

        if not (has_rx1 and has_rx2):
            logging.info("Skipping '%s' (missing real_xray1/2)", in_path)
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Create output file and copy top-level members, renaming the two
        # xray datasets to the new names. We avoid copying any input item that
        # would collide with the target names (xray1/xray2) because those are
        # created from real_xray1/2. Copy operations are guarded so a failure
        # on one member doesn't abort copying of others.
        target_names = {"xray1", "xray2"}

        with h5py.File(out_path, "w") as fout:
            for name in fin:
                try:
                    if name == "real_xray1":
                        logging.debug("Renaming real_xray1 -> xray1 in %s", out_path)
                        fin.copy("real_xray1", fout, name="xray1")
                    elif name == "real_xray2":
                        logging.debug("Renaming real_xray2 -> xray2 in %s", out_path)
                        fin.copy("real_xray2", fout, name="xray2")
                    else:
                        # If the input already contains xray1/xray2, skip them so
                        # they don't overwrite the renamed datasets we just added.
                        if name in target_names:
                            logging.debug(
                                "Skipping input name '%s' to avoid collision in %s",
                                name,
                                out_path,
                            )
                            continue

                        # copy item with same name
                        fin.copy(name, fout)
                except Exception as exc:  # broad catch to keep processing other items
                    logging.warning(
                        "Failed to copy '%s' from %s -> %s: %s",
                        name,
                        in_path,
                        out_path,
                        exc,
                    )

        logging.info("Wrote patched file: %s", out_path)


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

    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 2

    found = list(find_ct_files(input_dir))
    logging.info("Found %d ct_xray_data.h5 files under %s", len(found), input_dir)

    processed = 0
    skipped = 0

    for in_file in found:
        # preserve directory structure relative to input_dir
        try:
            rel = in_file.relative_to(input_dir)
        except Exception:
            # fallback: use file name only
            rel = in_file.name

        out_file = output_dir.joinpath(rel)

        try:
            process_file(in_file, out_file)
        except Exception as exc:
            logging.exception("Error processing '%s': %s", in_file, exc)
            skipped += 1
            # try to remove a possibly partially-written file
            try:
                if out_file.exists():
                    out_file.unlink()
            except Exception:
                pass
            continue

        # check whether we wrote output and that it contains expected datasets
        try:
            with h5py.File(out_file, "r") as fout:
                if "xray1" in fout and "xray2" in fout:
                    processed += 1
                else:
                    logging.warning(
                        "Output file %s does not contain xray1/xray2", out_file
                    )
                    skipped += 1
        except Exception:
            skipped += 1

    logging.info("Completed. Processed: %d, Skipped: %d", processed, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
