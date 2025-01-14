"""
Iterates over all dicom files and creates a csv with all dicom tags
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import json

import pydicom
from tqdm import tqdm

# Define tags to exclude from the CSV based on their keywords
EXCLUDED_TAGS_KEYWORDS = ["PixelData", "PixelRepresentation"]


def read_dicom_tags(dicom_file: Path) -> tuple[dict, dict]:
    try:
        dicom = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        tags_dict = {
            str(tag): dicom[tag].value
            for tag in dicom.keys()
            if dicom[tag].keyword not in EXCLUDED_TAGS_KEYWORDS
        }
        tags_dict["Filepath"] = str(dicom_file)  # Add file path to tags dictionary

        keywords_dict = {
            str(tag): dicom[tag].keyword or str(tag)
            for tag in dicom.keys()
            if dicom[tag].keyword not in EXCLUDED_TAGS_KEYWORDS
        }

        return tags_dict, keywords_dict
    except Exception as e:
        print(f"Error reading {dicom_file}: {e}")
        return {
            "Filepath": str(dicom_file)
        }, {}  # Include file path even in case of error


def get_dicom_files(start_path: Path) -> list[Path]:
    dicom_files = []
    for root, dirs, files in os.walk(start_path, followlinks=True):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(Path(root) / file)
    return dicom_files


def write_dicom_tags_to_csv(
    dicom_tags_results: list[tuple],
    all_tags: set,
    output_dir="preprocess",
    output_filename="dicom_tags.csv",
):
    csv_filepath = Path(output_dir) / output_filename
    csv_filepath.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the directory exists

    with open(csv_filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_tags, delimiter=";")
        writer.writeheader()

        for (
            tags_dict,
            _,
        ) in (
            dicom_tags_results
        ):  # assuming dicom_tags_results contains tuples of (tags_dict, keywords_dict)
            # Fill missing tags with 'NaN' to handle new tags across files
            filled_tags = {tag: tags_dict.get(tag, "NaN") for tag in all_tags}
            writer.writerow(filled_tags)

    print(f"Tags from DICOM files have been written to {csv_filepath}")


def write_tags_keywords_to_json(
    dicom_tags_results: list[tuple],
    output_dir="preprocess",
    output_filename="dicom_keywords.json",
):
    # Combine keywords from all files
    all_keywords = {}
    for _, keywords_dict in dicom_tags_results:
        all_keywords.update(keywords_dict)
    all_keywords["Filepath"] = "Filepath"  # adds the filepath column for mapping

    json_filepath = Path(output_dir) / output_filename
    json_filepath.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the directory exists

    with open(json_filepath, "w") as jsonfile:
        json.dump(all_keywords, jsonfile, indent=4)

    print(f"Tags to keywords dictionary have been written to {json_filepath}")


def main(directory_path):
    path = Path(directory_path)
    if not path.is_dir():
        raise ValueError(f"{directory_path} is not a valid directory")

    print("Looking for dicom files inside directory tree...")
    # Find all DICOM files in the directory tree
    dicom_files = get_dicom_files(path)
    print(f"Found {len(dicom_files)} dicom files inside directory tree")

    print("Accessing dicoms and retrieving tags...")
    # Use ThreadPoolExecutor to read DICOM files concurrently
    with ThreadPoolExecutor() as executor:
        dicom_tags_results = list(
            tqdm(executor.map(read_dicom_tags, dicom_files), total=len(dicom_files))
        )
    print("Retrieved all tags present in all dicoms!")

    # Combine all tags and keywords from all files
    all_tags = set()
    all_keywords = set()
    # iterates over results, each item is a tuple (tags_dict, keyword_dict)
    for tags_dict, keywords_dict in dicom_tags_results:
        # updates the set to also include all the keys from that dictionary
        all_tags.update(tags_dict.keys())
        all_keywords.update(keywords_dict.values())

    print("Writing results...")
    # Write to CSV, using tags as headers
    write_dicom_tags_to_csv(dicom_tags_results, all_tags)
    # Writes a dictionary to map tags (numbers) to keywords
    write_tags_keywords_to_json(
        dicom_tags_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DICOM tags to CSV")
    parser.add_argument(
        "path", type=str, help="Path to the directory containing DICOM files"
    )
    # parser.add_argument("--output-name", type=str, help="The output name to be used on the files")
    args = parser.parse_args()

    main(args.path)
