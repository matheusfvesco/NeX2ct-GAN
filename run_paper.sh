#!/usr/bin/env bash
set -euo pipefail

# Optional metadata refresh
# python paper-figure-scripts/generate-tags-csv.py data

# 1) Build the complete dataset (synthetic inputs + real_xray fields)
python preprocess.py

# 2) Create patched datasets:
#    - real-only: ...-real
#    - mixed counterpart written automatically by patch_dataset as ...-real_mixed
python paper-figure-scripts/patch_dataset.py \
	output/preprocess/GAN-dataset-complete-clipped-shifted \
	output/preprocess/GAN-dataset-complete-clipped-shifted-real

# 3) Generate split files.
#    Synthetic and real are generated first.
python paper-figure-scripts/split_samples.py \
	output/preprocess/GAN-dataset-complete-clipped-shifted \
	splits synthetic 0.1

python paper-figure-scripts/split_samples.py \
	output/preprocess/GAN-dataset-complete-clipped-shifted-real \
	splits real 0.1

#    Mixed split is derived from synthetic+real split files. The mixed branch
#    also overwrites synthetic splits to match mixed splits exactly.
python paper-figure-scripts/split_samples.py \
	output/preprocess/GAN-dataset-complete-clipped-shifted-real_mixed \
	splits mixed 0.1

# Optional previews/figures
# python paper-figure-scripts/preview-processed-dataset.py \
#     output/preprocess/GAN-dataset-complete-clipped-shifted \
#     output/preview/data/GAN-dataset-complete-clipped-shifted
#
# python paper-figure-scripts/preview-processed-dataset.py \
#     output/preprocess/GAN-dataset-complete-clipped-shifted-real \
#     output/preview/data/GAN-dataset-complete-clipped-shifted-real
#
# python paper-figure-scripts/generate-drr-processing-figures.py \
#     data output/processing-examples --full-unordered-samples