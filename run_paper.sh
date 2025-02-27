#python paper-figure-scripts/generate-tags-csv.py data

python preprocess.py

python paper-figure-scripts/split_samples.py preprocess/GAN-dataset splits 0.05

python paper-figure-scripts/preview_gan_dataset.py

python paper-figure-scripts/generate-drr-processing-figures.py data output/processing-examples --full-unordered-samples