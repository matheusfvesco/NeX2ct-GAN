# harpia
Codes for the Harpia XR-CT project

## Folder structure

Your folder should look like this:

```
harpia
├── data # all folders should contain folders named after patient id, inside the ones listed here
│   ├── LIDC-IDRI # should contain directories for each patient id
│   ├── MIDRC-RICORD-1A
|   ├── MIDRC-RICORD-1C 
│   └── xrays-view-annotation.json # json file with x-ray view annotations used in preprocess
├── lib
│   ├── data
│   ├── dicom
│   ├── io
│   ├── model
│   ├── process
│   └── utils
├── models # folder where the models should be stored
│   ├── classification
│   └── gan
├── output # folder where outputs will be saved
│   └── preprocess
│       └── GAN-dataset # processed dataset
```

Ensure all of the datasets are properly downloaded from TCIA. Inside each dataset folder, there should be directories named after the patient ids (e.g., `data/LIDC-IDRI/LIDC-IDRI-0001`, `data/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000082`).

Download the model checkpoints and place them under `models/gan/best.pt`

## Instalation

We recommend that you start from the `nvcr.io/nvidia/pytorch:23.07-py3` container image, or use the .devcontainer. For this, ensure that you have docker and the nvidia container toolkit installed on your machine.

Then use
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data preprocessing

#### 1.0. (optional) Checking x-ray annotations or annotating them 

Some datasets don't have annotations for the x-ray views. You can either check the annotations provided with the repository (`data/xrays-view-annotation.json`) or manually annotate them.

To check the annotations, use:
```bash
python manually-annotate-xray-views.py --datasets "gan_dataset" --root "data" --savepath "output/temporary-files/xrays-manual-annotation" --check
```
This will save jpeg thumbnails of the views inside the specified savepath using the already provided json file. You can inspect the folders `frontal` and `lateral` and check if the views are correctly annotated.

To annotate, use:
```bash
python manually-annotate-xray-views.py --datasets "gan_dataset" --root "data" --savepath "output/temporary-files/xrays-manual-annotation"
```

This will create 2 folders inside the savepath: `frontal` and `lateral`. Inside each folder, jpeg thumbnails of the x-ray views will be created using the existent annotation in the tags. Alongside the folders, thumbnails of the views without annotations will be created. Simply drag the thumbnails to their correct folder. Then run:
```bash
python generate-xray-annotations.py
```
This should create a json file inside the data folder with all the annotations for each x-ray view in the dataset.

#### 1.1. Dataset Preparation

To preprocess the dataset, run
```bash
python preprocess.py
```

This runs inventory, selection, and generation in one invocation. The audit is written to `output/preprocess/GAN-dataset-complete/audit/exams.json` and `run.json`. By default, the first blocking error stops processing after saving the audit. Use `--continue-on-error` to attempt independent remaining cases; the command still exits with an error if any case failed.

To run a deterministic subset while sampling independently from every dataset index, use `--subset` (3 patients per dataset) or `--subset X` (up to X unique patients per dataset):

```bash
python preprocess.py --subset --output-root output/preprocess/GAN-dataset-subset
python preprocess.py --subset 5 --output-root output/preprocess/GAN-dataset-subset-5
```

The selection uses a fixed SHA-256 ranking and is recorded in `run.json`. It is generated in memory for that invocation; split files are not changed.
Patients outside the sample are audited as out of scope without opening their DICOMs. Subset success only validates the sampled cohort, not the complete dataset; `run.json` identifies the execution scope explicitly.

**Always give subset runs a separate `--output-root`.** The default `output/preprocess/GAN-dataset-complete` is reserved for a full run. A subset writes only its sampled patients, so using that default can leave it looking like the complete dataset. The tests in `tests/test_datasets.py` check every patient in the full splits and will fail against subset output. Run `python preprocess.py` to populate the complete output before running those tests.

`--output-root` must be outside `--data-root`: the raw dataset is input only. If inventory is interrupted, `run.json` lists unverified patients separately from confirmed missing patients. Clinical export can read the consolidated report directly:

```bash
python paper-figure-scripts/export-clinical-audit.py \
  --run output/preprocess/GAN-dataset-complete/audit/run.json \
  --output-dir output/clinical-audit
```

This should create multiple h5 files located at `output/preprocess/GAN-dataset-complete`, following the structure `patient_id / ct_xray_data.h5`.

#### 1.2. Dataset patching

To create the real data only version of the dataset, run
```bash
python paper-figure-scripts/patch_dataset.py output/preprocess/GAN-dataset-complete output/preprocess/GAN-dataset-complete-real
```

This should create a new version of the dataset located at `output/preprocess/GAN-dataset-complete-real`.

#### 1.3. (optional) Dataset preview

To preview the datasets, use the following commands:

```bash
python3 paper-figure-scripts/preview-processed-dataset.py \
    output/preprocess/GAN-dataset-complete \
    output/preview/data/GAN-dataset-complete
```
This should create jpeg visualizations of the complete dataset

Run
```bash
python3 paper-figure-scripts/preview-processed-dataset.py \
    output/preprocess/GAN-dataset-complete-real \
    output/preview/data/GAN-dataset-complete-real
```
To preview the patched dataset inside the directory `output/preview/data/GAN-dataset-complete-real`

#### 1.4. Dataset splitting

Run the following commands to create dataset splits inside the `splits/` directory
```bash
python paper-figure-scripts/split_samples.py output/preprocess/GAN-dataset-complete splits synthetic 0.1
```

```bash
python paper-figure-scripts/split_samples.py output/preprocess/GAN-dataset-complete-real splits real 0.1
```
