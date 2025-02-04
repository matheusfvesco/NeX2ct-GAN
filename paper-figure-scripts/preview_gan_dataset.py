"""
Creates previews for the entire preprocessed dataset
"""
from lib.io import HDF5FileHandler
from lib.utils.viz import ArrayDisplayUtility
from pathlib import Path
from tqdm import tqdm

GAN_DATASET = Path("preprocess/GAN-dataset")
OUTPUT_PATH = GAN_DATASET / "preview"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

exclude_tags = ["metadata", "ct"]

for path in tqdm(list(GAN_DATASET.rglob("*.hdf5"))):
    handler = HDF5FileHandler(path, mode="read_only")

    # ['ct', 'frontal_drr', 'frontal_xray', 'lateral_drr', 'lateral_xray', 'metadata']
    for key in handler.file.keys():
        if key in exclude_tags:
            continue
        array = handler.retrieve_array(key)
        visualizer = ArrayDisplayUtility(array)
        visualizer.write_as_image(
            OUTPUT_PATH / f"{path.stem}_{key}.jpeg", thumbnail=200
        )

    array = handler.retrieve_array("ct")
    visualizer = ArrayDisplayUtility(array)
    visualizer.write_as_image(
        OUTPUT_PATH / f"{path.stem}_ct_0.jpeg",
        thumbnail=200,
        axis=0,
        complete_array=False,
    )
    visualizer.write_as_image(
        OUTPUT_PATH / f"{path.stem}_ct_1.jpeg",
        thumbnail=200,
        axis=1,
        complete_array=False,
    )
    visualizer.write_as_image(
        OUTPUT_PATH / f"{path.stem}_ct_2.jpeg",
        thumbnail=200,
        axis=2,
        complete_array=False,
    )
