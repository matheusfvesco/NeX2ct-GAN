"""
Script that uses the output of manually-annotate-xray-views.py to generate
a json file that is used by some of the lib classes.
"""

import argparse
import json
from pathlib import Path

def process_directory(root_path: Path):
    """
    For each jpeg file stored in each class directory ('frontal' or
    'lateral'), creates a dictionary where each key is the complete
    relative path to a dicom file
    """
    classes = ["excluded", "selected"]
    data = {}

    for class_name in classes:
        class_path = root_path / class_name
        if not class_path.exists() or not class_path.is_dir():
            raise ValueError(f"{class_path} doesn't exist or is not a directory")
        for file in class_path.glob("*-ct.jpeg"):
            # Remove the .jpeg suffix
            patient_id = file.stem.replace("-ct", "")

            # adding the patient_id as the key and the value its class
            data[patient_id] = class_name

    return data


def write_json(data: dict, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "patient-selection-annotation.json"

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
