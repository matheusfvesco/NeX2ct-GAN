"""
Generate figures illustrating the preprocessing pipeline for DRR (Digitally Reconstructed Radiograph) generation.

This script processes a subset of CT scans, applies various processing steps to each scan,
and generates images visualizing the results. The output is saved in a directory structure
with subdirectories representing different processing orders and unordered versions.
"""

import sys
import os

# sys.path.insert(1, os.path.abspath("."))

from lib.dicom import (
    CTScanDirectory,
    SingleViewXRayDirectory,
    MultiviewXRayDirectory,
)
from lib.process.drr import DRRGenerator
from lib.utils.viz import ArrayDisplayUtility
from lib.data.dicom import DicomDataset
from lib.logging import setup_logger

from pathlib import Path
import random
import csv
import argparse
from tqdm import tqdm


def main(args):
    random.seed(args.seed)
    DATA_ROOT = Path(args.input)
    SAVE_ROOT = Path(args.output)
    main_progress_bar = tqdm(range(4))
    THUMBNAIL_SIZE = args.thumbnail_size if args.thumbnail_size else None
    FULL_UNORDERED = args.full_unordered_samples
    FIRST_AND_LAST_UNORDERED = args.first_and_last_unordered_samples

    logger = setup_logger(
        logger_name="preprocess",
        experiment_name="Generating DRR figures",
        filename_comment="drr_figures",
    )

    logger.info("Creating dataset dictionary...")
    main_progress_bar.set_description("Creating dataset dictionary...")
    dataset = DicomDataset(root_dir=DATA_ROOT, logger=logger)
    dataset.load()
    main_progress_bar.update()

    main_progress_bar.set_description(
        "Filtering invalid CTs and sorting by the amount of slices..."
    )
    dataset.filter_dataset()

    main_progress_bar.update()

    selected_patients = dataset.get_demo_subset_names(
        seed=args.seed,
        sample_size_porcentage=args.sample_size,
        ensure_upside_down_cases=True,
    )

    csv_data = []

    logger.info("Generating processing pipeline demos...")
    main_progress_bar.set_description("Generating processing pipeline demos...")
    dataset_bar = tqdm(selected_patients)
    for patient in dataset_bar:
        logger.debug(f"Creating demo for patient: {patient}")
        dataset_bar.set_description(patient)
        ct_scans: list[CTScanDirectory] = dataset[patient]["cts"]
        singleview_dirs: list[SingleViewXRayDirectory] = dataset[patient]["xrays"]
        multiview_dirs: list[MultiviewXRayDirectory] = dataset[patient]["multiview"]

        processing_version = [
            ("", True)
        ]  # first is suffix to the folder name, and second is wether z_ordering is used or not.

        if FULL_UNORDERED:
            processing_version.append(("-unordered", False))

        if FIRST_AND_LAST_UNORDERED:
            pass

        logger.debug(
            f"FIRST_AND_LAST_UNORDERED: {FIRST_AND_LAST_UNORDERED} | FULL_UNORDERED: {FULL_UNORDERED}"
        )

        if not ct_scans:
            continue

        # the complete preprocess order.
        custom_process_order = [
            "set_min_to_max",
            "raw_array_to_hu",
            "rescale",
            "set_max_to_min",
            "normalize",
        ]

        # generates a list where each item contains the previous item in
        # the new list + the next item in the old complete list.
        # basically a list where each item is a step of the preprocess
        # order + the previous steps also applied
        custom_process_orders_steps = [
            custom_process_order[: i + 1] for i in range(len(custom_process_order))
        ]

        for folder_suffix, use_z_ordering in processing_version:
            # writes all of the processing steps
            for idx, step in enumerate(custom_process_orders_steps):
                folder_name = str(idx + 2) + ("-".join(step)) + folder_suffix
                if (SAVE_ROOT / folder_name / f"{patient}-frontal.jpeg").exists():
                    continue

                # applies min-max normalization to 0-255 range
                custom_process_order = step
                array = (
                    ct_scans[0]
                    ._get_processor(use_z_ordering=use_z_ordering)
                    ._process_compose(custom_process_order)
                )
                array_visualizer = ArrayDisplayUtility(array)
                array_visualizer.write_as_image(
                    path=SAVE_ROOT / folder_name / f"{patient}-scan.gif",
                    complete_array=True,
                    thumbnail=THUMBNAIL_SIZE,
                )
                array_visualizer.write_as_image(
                    path=SAVE_ROOT / folder_name / f"{patient}-scan.jpeg",
                    complete_array=False,
                    thumbnail=THUMBNAIL_SIZE,
                )
                range_val = array_visualizer.range
                precision_val = array_visualizer.precision
                csv_data.append([patient, folder_name, range_val, precision_val])
                logger.debug(
                    f"{folder_name}: {array_visualizer.range}, {array_visualizer.precision}"
                )
                drr_generator = DRRGenerator(
                    ct_scans[0],
                    custom_process_order=custom_process_order,
                    use_z_ordering=use_z_ordering,
                )
                frontal = drr_generator.frontal_view
                lateral = drr_generator.lateral_view
                ArrayDisplayUtility(frontal).write_as_image(
                    SAVE_ROOT / folder_name / f"{patient}-frontal.jpeg"
                )
                ArrayDisplayUtility(lateral).write_as_image(
                    SAVE_ROOT / folder_name / f"{patient}-lateral.jpeg"
                )

        for folder_suffix, use_z_ordering in processing_version:
            # ----------------------------------------------
            # (1) no preprocessing at all
            custom_process_order = ["raw_array_to_hu"]
            folder_name = str(1) + ("-".join(custom_process_order)) + folder_suffix
            logger.debug(f"{folder_name}")
            if not (SAVE_ROOT / folder_name / f"{patient}-scan.gif").exists():
                array = (
                    ct_scans[0]
                    ._get_processor(use_z_ordering=use_z_ordering)
                    ._process_compose(custom_process_order)
                )
                array_visualizer = ArrayDisplayUtility(array)
                array_visualizer.write_as_image(
                    path=SAVE_ROOT / folder_name / f"{patient}-scan.gif",
                    complete_array=True,
                    thumbnail=THUMBNAIL_SIZE,
                )
                array_visualizer.write_as_image(
                    path=SAVE_ROOT / folder_name / f"{patient}-scan.jpeg",
                    complete_array=False,
                    thumbnail=THUMBNAIL_SIZE,
                )
                range_val = array_visualizer.range
                precision_val = array_visualizer.precision
                csv_data.append([patient, folder_name, range_val, precision_val])
                logger.debug(
                    f"{folder_name}: {array_visualizer.range}, {array_visualizer.precision}"
                )
                drr_generator = DRRGenerator(
                    ct_scans[0],
                    custom_process_order=custom_process_order,
                    use_z_ordering=use_z_ordering,
                )
                frontal = drr_generator.frontal_view
                lateral = drr_generator.lateral_view
                ArrayDisplayUtility(frontal).write_as_image(
                    SAVE_ROOT / folder_name / f"{patient}-frontal.jpeg"
                )
                ArrayDisplayUtility(lateral).write_as_image(
                    SAVE_ROOT / folder_name / f"{patient}-lateral.jpeg"
                )

        if FIRST_AND_LAST_UNORDERED:
            # ---------------------------------------------------
            # (0) No preprocessing and Doesn't read in the correct order
            custom_process_order = ["raw_array_to_hu"]
            folder_name = str(0) + ("-".join(custom_process_order)) + "-unordered"
            if not (SAVE_ROOT / folder_name / f"{patient}-scan.gif").exists():
                array = (
                    ct_scans[0]
                    ._get_processor(use_z_ordering=False)
                    ._process_compose(custom_process_order)
                )
                array_visualizer = ArrayDisplayUtility(array)
                array_visualizer.write_as_image(
                    path=SAVE_ROOT / folder_name / f"{patient}-scan.gif",
                    complete_array=True,
                    thumbnail=THUMBNAIL_SIZE,
                )
                array_visualizer.write_as_image(
                    path=SAVE_ROOT / folder_name / f"{patient}-scan.jpeg",
                    complete_array=False,
                    thumbnail=THUMBNAIL_SIZE,
                )
                range_val = array_visualizer.range
                precision_val = array_visualizer.precision
                csv_data.append([patient, folder_name, range_val, precision_val])
                logger.debug(
                    f"{folder_name}: {array_visualizer.range}, {array_visualizer.precision}"
                )
                drr_generator = DRRGenerator(
                    ct_scans[0],
                    custom_process_order=custom_process_order,
                    use_z_ordering=False,
                )
                frontal = drr_generator.frontal_view
                lateral = drr_generator.lateral_view
                ArrayDisplayUtility(frontal).write_as_image(
                    SAVE_ROOT / folder_name / f"{patient}-frontal.jpeg"
                )
                ArrayDisplayUtility(lateral).write_as_image(
                    SAVE_ROOT / folder_name / f"{patient}-lateral.jpeg"
                )

            # ---------------------------------------------------------
            # (7) unordered with all processing steps
            custom_process_order = [
                "set_min_to_max",
                "raw_array_to_hu",
                "rescale",
                "set_max_to_min",
                "normalize",
            ]
            folder_name = (
                str(len(custom_process_order) + 2)
                + ("-".join(custom_process_order))
                + "-unordered"
            )

            if not (SAVE_ROOT / folder_name / f"{patient}-scan.gif").exists():
                array = (
                    ct_scans[0]
                    ._get_processor(use_z_ordering=False)
                    ._process_compose(custom_process_order)
                )
                array_visualizer = ArrayDisplayUtility(array)
                array_visualizer.write_as_image(
                    path=SAVE_ROOT / folder_name / f"{patient}-scan.gif",
                    complete_array=True,
                    thumbnail=THUMBNAIL_SIZE,
                )
                array_visualizer.write_as_image(
                    path=SAVE_ROOT / folder_name / f"{patient}-scan.jpeg",
                    complete_array=False,
                    thumbnail=THUMBNAIL_SIZE,
                )
                range_val = array_visualizer.range
                precision_val = array_visualizer.precision
                csv_data.append([patient, folder_name, range_val, precision_val])
                logger.debug(
                    f"{folder_name}: {array_visualizer.range}, {array_visualizer.precision}"
                )

                # drr generator handles processing inside the class, so we need to pass
                # the arguments again, just like we did in the previous step.
                drr_generator = DRRGenerator(
                    ct_scans[0],
                    custom_process_order=custom_process_order,
                    use_z_ordering=False,
                )
                frontal = drr_generator.frontal_view
                lateral = drr_generator.lateral_view
                ArrayDisplayUtility(frontal).write_as_image(
                    SAVE_ROOT / folder_name / f"{patient}-frontal.jpeg"
                )
                ArrayDisplayUtility(lateral).write_as_image(
                    SAVE_ROOT / folder_name / f"{patient}-lateral.jpeg"
                )

        logger.debug(f"Finished processing {patient}\n")

        ct_scans[0].cache.clear()

        if not (singleview_dirs or multiview_dirs):
            continue
        counter = 0
        for singleview_dir in singleview_dirs:
            singleview_dir[0].visualizer.write_as_image(
                SAVE_ROOT / "xrays" / f"{patient}-singleview{counter}.jpeg",
                thumbnail=THUMBNAIL_SIZE,
            )
            counter += 1
        for multiview_dir in multiview_dirs:
            for view in multiview_dir:
                if not view.is_valid:
                    continue
                view.visualizer.write_as_image(
                    SAVE_ROOT / "xrays" / f"{patient}-multiview{counter}.jpeg",
                    thumbnail=THUMBNAIL_SIZE,
                )
                counter += 1
    dataset_bar.close()
    main_progress_bar.update()

    csv_filename = "processing_order_data.csv"
    csv_filepath = SAVE_ROOT / csv_filename

    main_progress_bar.set_description("Writing CSV file...")
    # Writing data to CSV using ';' as delimiter
    with open(csv_filepath, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Patient", "Processing Order", "Range", "Precision"])  # Header
        writer.writerows(csv_data)

    main_progress_bar.update()

    logger.info("Success!")
    logger.info(f"All the preprocessing demonstrations have been saved to {SAVE_ROOT}")
    logger.info(f"CSV file '{csv_filename}' has been created at: {csv_filepath}")
    main_progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Generates the
                                     preprocessing pipeline examples for
                                     a subset of the dataset"""
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input file path.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="""Output file path. 
        This should be a directory where multiple subdirectories will be
        created to store each step. A CSV file will also be created inside
        this directory, containing information like the patient, the
        preprocessing order, the range of values in the array, and the
        precision (fp32, fp64, etc.) of the array""",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="Random seed for reproducibility. Defaults to 40. We used this seed because a few of the selected CTs do actually require ordering the slices by the slice Z position relative to the patient.",
        default=40,
    )
    parser.add_argument(
        "--sample-size",
        "-S",
        type=int,
        help="""The porcentage of scans 
        that will be selected to be used for generating examples. Defaults to 10""",
        default=10,
    )
    parser.add_argument(
        "--thumbnail-size",
        "-ts",
        type=int,
        help="""The size of the thumbnails that will be generated for each CT. Defaults to 200. If set to 0, full size will be used to save the files.""",
        default=200,
    )
    parser.add_argument(
        "--full-unordered-samples",
        action="store_true",
        help="For each directory, also saves a version of what happens when z is not used for ordering the CTs.",
    )
    parser.add_argument(
        "--first-and-last-unordered-samples",
        action="store_true",
        help="Doesn't save examples before and after the full processing pipeline of what happens when z is not used for ordering the CTs.",
    )
    args = parser.parse_args()
    main(args=args)
