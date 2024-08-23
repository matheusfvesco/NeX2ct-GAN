"""
This selects a few demonstration images from the dataset and copies them
to the specified directory, in order to allow easier access of the examples
provided in
"""

import os
import shutil
import argparse
from pathlib import Path


def main(args):
    source_folder = Path(args.source_folder)
    destination_folder = Path(args.destination_folder)
    prefixes = args.prefixes

    for root, _, files in os.walk(source_folder, followlinks=True):
        for file in files:
            for prefix in prefixes:
                if file.startswith(prefix):
                    src_file_path = Path(root) / file

                    relative_path = src_file_path.relative_to(source_folder)
                    dest_file_path = destination_folder / prefix / relative_path

                    dest_dir = dest_file_path.parent

                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file_path, dest_file_path, follow_symlinks=True)
                    print(f"Copied {src_file_path} to {dest_file_path}")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy files with specified prefixes while preserving directory structure."
    )
    parser.add_argument(
        "source_folder", type=str, help="The source folder to search for files."
    )
    parser.add_argument(
        "destination_folder", type=str, help="The destination folder to copy files to."
    )
    parser.add_argument(
        "prefixes", type=str, nargs="+", help="List of prefixes to search for."
    )

    args = parser.parse_args()
    main(args)
