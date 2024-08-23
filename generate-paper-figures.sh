#!/bin/bash

# Paths for script, data, and output directories
SCRIPTS_FOLDER="./paper-figure-scripts"
DATA_FOLDER="./data"
OUTPUT_FOLDER="./output"

#python "$SCRIPTS_FOLDER/generate-tags-csv.py" "$DATA_FOLDER"
#jupyter nbconvert --to notebook --execute "$SCRIPTS_FOLDER/analyze-dataset.ipynb" --inplace

# generates 
python "$SCRIPTS_FOLDER/generate-drr-processing-figures.py" "$DATA_FOLDER" "$OUTPUT_FOLDER/preprocess/preprocessing-order-demo" "--full-unordered-samples" # "--first-and-last-unordered-samples"


# selects a few of the cases in which z ordering is important for the DRR generation, and sets them apart for easy use.
python "$SCRIPTS_FOLDER/select-preprocessing-demos.py" "$OUTPUT_FOLDER/preprocess/preprocessing-order-demo/" "$OUTPUT_FOLDER/preprocess/selected-preprocessing-demo" "LIDC-IDRI-0052" "LIDC-IDRI-0091" "LIDC-IDRI-0249" "419639-000831"