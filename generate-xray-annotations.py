"""
Script that uses the output of manually-annotate-xray-views.py to generate
a json file that is used by some of the lib classes.
"""

import argparse
import json
from pathlib import Path


def filename_to_path(filename: str):
    """
    Converts a string representing a file path to a valid file path by:
    1. Removing the '.jpeg' extension.
    2. Replacing '@' characters with '/' for proper directory separation.

    Args:
        filename (str): A string in the format 'path@to@filename.dcm', such as 'MIDRC-RICORD-1C-SITE2-000148@04-09-2000-NA-NA-44042@1.000000-NA-80696@1-1.dcm.jpeg'.

    Returns:
        str: The modified file path in a valid format ('path/to/filename.dcm'), such as 'MIDRC-RICORD-1C-SITE2-000148/04-09-2000-NA-NA-44042/1.000000-NA-80696/1-1.dcm'.
    """
    return filename.replace(".jpeg", "").replace("@", "/")


def process_directory(root_path: Path):
    """
    For each jpeg file stored in each class directory ('frontal' or
    'lateral'), creates a dictionary where each key is the complete
    relative path to a dicom file
    """
    classes = ["frontal", "lateral"]
    data = {}

    for class_name in classes:
        class_path = root_path / class_name
        if not class_path.exists() or not class_path.is_dir():
            raise ValueError(f"{class_path} doesn't exist or is not a directory")
        for file in class_path.glob("*.jpeg"):
            # Remove the .jpeg suffix
            original_filename = file.name
            new_filename = filename_to_path(original_filename)

            # Adding the converted filename to the dictionary with its class
            # The key should be the filename without the .jpeg extension
            data[new_filename] = class_name

    return data


def write_json(data: dict, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "xrays-view-annotation.json"

    with output_file.open("w") as f:
        json.dump(data, f, indent=4)


def main(args):
    root_path = Path(args.root_path)
    data = process_directory(root_path)
    write_json(data, "data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process .dcm.jpeg files into a JSON classification."
    )
    parser.add_argument(
        "root_path",
        type=str,
        help="The root directory containing frontal and lateral folders.",
    )

    args = parser.parse_args()
    main(args=args)
