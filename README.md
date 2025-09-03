# LaBraM LoRA

EEG classification using LaBraM (Large Brain Model) with LoRA fine-tuning.

## Setup

### 1. Conda Environment
```bash
conda env create -f environment.yml
conda activate labram-env
```

### 2. Model Weights
Download `labram-base.pth` from [here](https://github.com/935963004/LaBraM/tree/main/checkpoints) and place it in:
```
weights/pretrained-models/labram-base.pth
```

### 3. Dataset
Place your HDF5 dataset file in:
```
data/dataset.h5
```

## Usage
```bash
python train.py
```

## Configuration
Edit `hyperparameters.yaml` to adjust model parameters, training settings, and dataset paths.
