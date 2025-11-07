"""
Generates thumbnails that can be used to manually annotate frontal and lateral x-rays

It first tries to find the annotations present inside the dicom tags, and prefills two folders with the thumbnails.
When the tag is not present or valid, it creates the thumbnail outside those two folders, inside the root directory.
"""

from pathlib import Path
import os

from lib.dicom import DicomDirectory
from tqdm import tqdm
import argparse
from pydicom import dcmread


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="LIDC-IDRI,MIDRC-RICORD-1A,MIDRC-RICORD-1C",
        help="The datasets used to process, comma separated.",
    )
    parser.add_argument(
        "--root", type=str, default="data", help="The root directory of the dataset."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="output/temporary-files/xrays-manual-annotation",
        help="The path where the temporary files will be saved",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="If provided, it will use the already provided json file to retrieve the annotations",
    )
    return parser.parse_args()


def main(args):
    DATA_ROOT = Path(args.root)
    DATASETS = args.datasets.split(",")

    SAVE_PATH = Path(args.savepath)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    # if argument provided, this will be true, and use the provided json to get the annotations instead
    CHECK = args.check

    def path_to_filename(path: str):
        """
        Parses path to a filename that can be written to disk by replacing
        slashes with '@'
        """
        return str(path).replace("/", "@")

    def get_view_type(path: str):
        mappings = {
            "AP": "frontal",
            "PA": "frontal",
            "LL": "lateral",
            "LAT": "lateral",
        }
        dcm = dcmread(path)
        view_type = dcm.get("ViewPosition", "")
        if not view_type:
            return None
        for mapping in mappings:
            if mapping in view_type:
                return mappings[mapping]
        raise ValueError(f"'{view_type}' is not a valid view_type")

    total = [
        (dirpath, dirnames, filenames)
        for dataset_name in DATASETS
        for (dirpath, dirnames, filenames) in os.walk(DATA_ROOT / dataset_name)
    ]

    print("Reading the datasets...")

    progress_bar = tqdm(total=len(total))

    for dataset in DATASETS:
        dataset_path = DATA_ROOT / dataset
        tqdm.write(str(dataset_path))
        progress_bar.set_description(str(dataset_path))
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            progress_bar.update()
            if not filenames or not any(
                filename.endswith(".dcm") for filename in filenames
            ):
                continue

            directory = DicomDirectory(dirpath)
            if directory.is_ct:
                continue

            for dicom_file in directory:
                # gets full path relative to the dataset root dir
                rel_dicom_path = dicom_file.dicom_path.relative_to(dataset_path)
                # converts the full path to a string acceptable as a filename on disk
                save_filename = path_to_filename(rel_dicom_path) + ".jpeg"
                # saves the files to disk
                if CHECK:
                    # if check, uses the provided json to get the annotations for the file.
                    view_type = dicom_file.view_type
                else:
                    # defaults to using the dicom tags instead
                    view_type = get_view_type(dicom_file.dicom_path)
                if view_type == "frontal":
                    image_save_path = SAVE_PATH / "frontal"
                    image_save_path.mkdir(parents=True, exist_ok=True)
                if view_type == "lateral":
                    image_save_path = SAVE_PATH / "lateral"
                    image_save_path.mkdir(parents=True, exist_ok=True)
                if view_type is None:
                    image_save_path = SAVE_PATH

                dicom_file.visualizer.write_as_image(image_save_path / save_filename)


if __name__ == "__main__":
    main(parse_args())
