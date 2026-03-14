# thesis-pipeline

Minimal, reproducible benchmark pipeline for MoNuSeg nuclei instance segmentation (inference-only).

## Scope
- Dataset: MoNuSeg only
- Models: Cellpose, StarDist, and a U-Net + Watershed baseline
- No training / fine-tuning
- Focus: accuracy + runtime + usability

## Local data (not committed)
Expected layout:

data/raw/
monuseg_train/
Tissue Images/ # .tif
Annotations/ # .xml
monuseg_test/
Tissue Images/ # .tif
Annotations/ # .xml

## Setup
```bash
conda env create -f environment.yml
conda activate thesis-pipeline

## Sanity check
python scripts/00_dataset_sanity.py
