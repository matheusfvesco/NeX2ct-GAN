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

To run data preprocessing:

```bash
python preprocess.py --input "data" --output "outputs/preprocess/gan-dataset"
```

#### 1.2. (optional) Checking x-ray annotations or annotating them 

Some datasets don't have annotations for the x-ray views. You can either check the annotations provided with the repository (`data/xrays-view-annotation.json`) or manually annotate them.

To check the annotations, use:
```bash
python manually-annotate-xray-views.py --datasets "LIDC-IDRI,MIDRC-RICORD-1A,MIDRC-RICORD-1C" --root "data" --savepath "output/temporary-files/xrays-manual-annotation" --check
```
This will save jpeg thumbnails of the views inside the specified savepath using the already provided json file. You can inspect the folders `frontal` and `lateral` and check if the views are correctly annotated.

To annotate, use:
```bash
python manually-annotate-xray-views.py --datasets "LIDC-IDRI,MIDRC-RICORD-1A,MIDRC-RICORD-1C" --root "data" --savepath "output/temporary-files/xrays-manual-annotation"
```

This will create 2 folders inside the savepath: `frontal` and `lateral`. Inside each folder, jpeg thumbnails of the x-ray views will be created using the existent annotation in the tags. Alongside the folders, thumbnails of the views without annotations will be created. Simply drag the thumbnails to their correct folder. Then run:
```bash
python generate-xray-annotations.py
```
This should create a json file inside the data folder with all the annotations for each x-ray view in the dataset.

#### 1.3. Dataset Preparation

To preprocess the dataset, run
```bash
python preprocess.py
```

This should create multiple h5 files located at `preprocess/GAN-dataset`.

### 2. Training

### 3. Testing

### 4. Inference
