#!/usr/bin/env python3
import argparse
from pathlib import Path
from tqdm import tqdm
from lib.io import HDF5FileHandler
from lib.utils.viz import ArrayDisplayUtility

def main(args):
    for patient_dir in tqdm(list(args.input_path.iterdir()), desc="Processing patients"):
        h5_file = patient_dir / "ct_xray_data.h5"
        if not h5_file.exists():
            continue

        patient = HDF5FileHandler(filepath=h5_file)
        keys = patient.file.keys()
        pid = patient_dir.name

        arrays = {
            "ct": patient.retrieve_array("ct"),
            "xray1": patient.retrieve_array("xray1"),
            "xray2": patient.retrieve_array("xray2"),
        }

        if "real_xray1" in keys:
            arrays["real_xray1"] = patient.retrieve_array("real_xray1")
        if "real_xray2" in keys:
            arrays["real_xray2"] = patient.retrieve_array("real_xray2")

        args.output_path.mkdir(parents=True, exist_ok=True)
        for name, arr in arrays.items():
            if name == "ct":
                visualizer = ArrayDisplayUtility(arr)
                for axis in range(3):
                    visualizer.write_as_image(
                        path=args.output_path / f"{pid}-{name}_axis{axis}.jpeg",
                        thumbnail=200,
                        axis=axis,
                        complete_array=False,
                    )
            else:
                ArrayDisplayUtility(arr).write_as_image(
                    path=args.output_path / f"{pid}-{name}.jpeg",
                    thumbnail=200,
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 patient data to images.")
    parser.add_argument("input_path", type=Path, help="Input directory containing patient folders")
    parser.add_argument("output_path", type=Path, help="Output directory to save images")
    args = parser.parse_args()
    main(args)
