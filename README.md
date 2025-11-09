## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [KU Data Preprocessing](#ku-data-preprocessing)
3. [LOSO Evaluation](#loso-evaluation)
4. [Running Experiments](#running-experiments)
5. [SLURM Job File](#slurm-job-file)
6. [Project Structure](#project-structure)
7. [Configuration Files](#configuration-files)
8. [Models](#models)
9. [Testing and Evaluation](#testing-and-evaluation)
10. [Data Format](#data-format)

---

## Installation and Setup

### Environment Setup

The project supports both GPU and CPU environments. Use the appropriate conda environment file:

```bash
# GPU environment (recommended for training)
conda env create -f enviroments/environment-gpu.yml
conda activate your-env

# CPU-only environment
conda env create -f enviroments/environment-cpu.yml
conda activate your-env
```

### Dependencies

Key dependencies include:
- PyTorch (with CUDA support for GPU)
- PEFT (for LoRA adapters)
- Transformers (for LaBraM model)
- h5py (for HDF5 data storage)
- wandb (for experiment tracking)
- scipy, numpy, pandas (for data processing)

---

## KU Data Preprocessing

The KU dataset preprocessing pipeline (`preprocessing/preprocess_KU_data.py`) is designed to prepare EEG motor imagery data for LaBraM and other models. The preprocessing includes several critical steps to ensure data quality and compatibility.

### Overview

The KU dataset contains EEG recordings from 54 subjects performing motor imagery tasks. Each subject has two sessions, with each session containing training and testing trials. The preprocessing pipeline:

1. Loads raw `.mat` files for each subject and session
2. Applies signal processing filters
3. Performs channel selection and alignment
4. Applies normalization (with leakage prevention)
5. Creates temporal patches for LaBraM
6. Saves preprocessed data in HDF5 format

### Detailed Preprocessing Steps

#### 1. **Data Loading**
- Loads `sess{XX}_subj{YY}_EEG_MI.mat` files from the source directory
- Extracts training and testing data from each session
- Concatenates sessions for each subject
- Shape: `(epochs, channels, time)` where epochs = train + test trials

#### 2. **Channel Selection**
- KU dataset has 62 channels
- Maps to standard 10-20 system channels
- Function: `select_channels()` matches available channels to target channels
- Only channels present in both datasets are retained

#### 3. **Signal Filtering**

**Bandpass Filter:**
- Default: 0.5-100 Hz (4th order Butterworth)
- Applied before downsampling to prevent aliasing
- Function: `apply_bandpass_filter()`

**Notch Filters:**
- Removes powerline noise at 50 Hz, 60 Hz, and 100 Hz
- Function: `apply_notch_filter()`

**Common Average Reference (CAR):**
- Re-references all channels to the average across channels
- Reduces common noise and artifacts
- Function: `apply_car()`

#### 4. **Downsampling**
- Original sampling rate: 1000 Hz
- Downsampling factor: 5 (default)
- Target sampling rate: 200 Hz
- Uses `scipy.signal.decimate()` for anti-aliasing

#### 5. **Temporal Patching (for LaBraM)**
- Splits each trial into non-overlapping temporal patches
- Default patch size: 200 samples (1 second at 200 Hz)
- For 800-sample trials: creates 4 patches
- Final shape: `(epochs, channels, num_patches, patch_size)`
- Example: `(400, 62, 4, 200)` for 400 trials, 62 channels, 4 patches of 200 samples

#### 6. **Normalization**

The preprocessing supports three normalization modes, all designed to prevent data leakage:

**A. Per-Subject Normalization (`mode="subject"`):**
- Computes mean and std per channel across all support trials
- Support trials = training trials (prevents leakage from test data)
- Applies these statistics to all trials (train + test)
- Stats shape: `(1, C)` where C = number of channels

**B. Per-Run Normalization (`mode="run"`):**
- Assumes 4 runs × 100 trials = 400 trials total
- Computes mean and std per channel, per run, using only support trials within each run
- Applies run-specific stats to all trials in that run
- Stats shape: `(R, C)` where R = number of runs


**C. Per-Trial Normalization (`mode="trial"`) - used in our experiments:**
- Instance normalization: each trial normalized independently
- Computes mean and std per channel for each trial
- Stats shape: `(E, C)` where E = number of epochs
- **Leakage-safe**: No cross-trial information used

#### 7. **Output Format**

Preprocessed data is saved in HDF5 format:
```
/s{subject_id}/
  ├── X: (epochs, channels, patches, patch_size) - float32
  ├── Y: (epochs,) - int64 (class labels: 0=right, 1=left)
  └── norm/
      ├── mu: normalization mean statistics
      └── std: normalization standard deviation statistics
```

### Running KU Preprocessing

```python
from preprocessing.preprocess_KU_data import DataPreprocessingConfig, preprocess_ku_data

# Using preset configuration for LaBraM
config = DataPreprocessingConfig.get_preset(
    preset_name="labram",
    source_path="data/raw/KU_raw",  # Path to raw .mat files
    target_path="data/preprocessed"  # Output directory
)

# Customize if needed
config.normalize_mode = "trial"  # or "subject" or "run"
config.patch_size = 200
config.downsample_factor = 5

# Run preprocessing
preprocess_ku_data(config)
```

Or from command line:
```bash
python preprocessing/preprocess_KU_data.py
```

### Preprocessing Configuration Options

The `DataPreprocessingConfig` class supports extensive customization:

```python
@dataclass
class DataPreprocessingConfig:
    # I/O
    source_path: str
    target_path: str
    output_name: str = "KU_mi_preprocessed.h5"
    
    # Sampling
    original_fs: int = 1000
    downsample_factor: int = 5
    
    # Filtering
    apply_bandpass: bool = True
    bandpass_low: float = 0.5
    bandpass_high: float = 100.0
    
    apply_notch: bool = True
    notch_frequencies: List[float] = [50.0, 60.0, 100.0]
    
    apply_car: bool = True
    
    # Normalization
    normalize_mode: str = "subject"  # "subject" | "run" | "trial"
    
    # Patching
    apply_patching: bool = True
    patch_size: int = 200
    patch_overlap: int = 0
    
    # Channel selection
    apply_channel_selection: bool = True
    selected_channels: List[str] = []  # Empty = use standard 10-20
```

---

## LOSO Evaluation

**LOSO (Leave-One-Subject-Out)** is a cross-validation strategy where one subject is held out as the test set, and the model is trained on all other subjects. This evaluates the model's ability to generalize to unseen subjects, which is critical for EEG-based brain-computer interfaces.

### LOSO Workflow

The LOSO evaluation process (`testing/LOSO.py` and `run_loso.py`) performs the following steps:

1. **For each subject (fold):**
   - Set the subject as the test set (`leave_out=[subject_id]`)
   - Train the model on all other subjects
   - Evaluate the trained model using few-shot testing on the held-out subject

2. **Few-Shot Testing:**
   - For each shot count (0, 1, 2, ..., N):
     - Sample N support trials per class from the test subject
     - Fine-tune the model on these support trials
     - Evaluate on remaining query trials
     - Repeat multiple times (n_repeats)

3. **Results Aggregation:**
   - Results saved per fold in `results/{model_folder_name}/{experiment_name}/test_results/`
   - Includes accuracy, loss, and other metrics for each shot count
   - Results can be aggregated across all folds for final analysis

### Running LOSO Evaluation

#### Command Line Interface

```bash
# Run LOSO for LaBraM with LoRA
python run_loso.py --model labram-lora

# Run LOSO for DeepConvNet
python run_loso.py --model deepconvnet

# Run LOSO for LaBraM with partial fine-tuning
python run_loso.py --model labram-partialft
```

The script automatically:
- Loads the appropriate configuration file (`hyperparameters/hyperparameters-{model}.yaml`)
- Iterates through all subjects (or subjects specified in config)
- Trains and evaluates each fold
- Saves results to `results/{model_folder_name}/`

#### Configuration

LOSO parameters are configured in the hyperparameter YAML files:

```yaml
test:
  n_epochs: 20              # Epochs for few-shot adaptation
  shots: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]  # Shot counts to evaluate
  n_repeats: 10             # Number of random repetitions per shot count
  model: "labram"           # Model type
  model_folder_name: "labram-lora"  # Output folder name

labram:
  test_lr: 0.00005          # Learning rate for few-shot adaptation
  test_wd: 0.05             # Weight decay for few-shot adaptation
```

#### Few-Shot Testing Details

The `TestEngine` class (`engines/test_engine.py`) implements the few-shot evaluation:

1. **Zero-Shot Evaluation:**
   - Evaluates model without any adaptation
   - Uses all test subject trials as query set
   - Provides baseline performance

2. **Few-Shot Evaluation (N > 0):**
   - Samples N support trials per class (balanced)
   - Remaining trials form the query set
   - Fine-tunes model on support set for `n_epochs`
   - Evaluates on query set
   - Repeats `n_repeats` times with different random seeds

3. **Cross-Validation for Adaptation:**
   - Optionally uses cross-validation to select the best adaptation epoch
   - Prevents overfitting to the support set

### Output Structure

```
results/
└── labram-lora/
    ├── labram-lora_fold001_subj1/
    │   ├── checkpoints/
    │   │   ├── best_model.pt
    │   │   └── final_model.pt
    │   └── test_results/
    │       ├── repetition_results.csv
    │       ├── aggregated_results.json
    │       └── plots/
    ├── labram-lora_fold002_subj2/
    │   └── ...
    └── ...
```

### Result Files

- `repetition_results.csv`: Detailed results for each repetition, shot count, and subject
- `aggregated_results.json`: Aggregated statistics (mean, std) across repetitions
- `plots/`: Visualization of few-shot learning curves

---

## Running Experiments

### Single-Run Training

For training a model on a single configuration:

```bash
python train.py
```

This uses the configuration in `hyperparameters/hyperparameters.yaml`. Modify the YAML file to change:
- Model type (labram, deepconvnet)
- Dataset path
- Hyperparameters
- Training settings

### LOSO Evaluation

As described above:
```bash
python run_loso.py --model labram-lora
```

### LOMSO (Leave-M-Subjects-Out)

For meta-learning or other cross-validation strategies:
```bash
python run_lomso.py
python run_lomso_meta.py  # For meta-learning
```

---

## SLURM Job File

The `labram-lora.job` file is a SLURM batch script for running experiments on HPC clusters (e.g., Snellius).

### Job File Contents

```bash
#!/bin/bash
#SBATCH --job-name=labram-train
#SBATCH --output=labram-train-%j.out
#SBATCH --error=labram-train-%j.err
#SBATCH --time=12:00:00           # walltime (adjust as needed)
#SBATCH --partition=gpu_h100      # GPU partition on Snellius
#SBATCH --gpus-per-node=1         # number of GPUs per node

# Load conda and activate environment
source ~/.bashrc
conda activate eeg2

# Set up W&B authentication
export WANDB_API_KEY=your_api_key_here
wandb login $WANDB_API_KEY

# Go to project folder and run training
cd ~/usirca/workspace/labram-lora
python run_loso.py --model labram-lora
```

### Job File Explanation

1. **SBATCH Directives:**
   - `--job-name`: Name for the job in the queue
   - `--output/--error`: Log files for stdout and stderr
   - `--time`: Maximum walltime (12 hours)
   - `--partition`: GPU partition name (adjust for your cluster)
   - `--gpus-per-node`: Number of GPUs requested


## Project Structure

```
labram-lora/
├── data/
│   ├── raw/                    # Raw datasets
│   │   ├── KU_raw/            # KU dataset .mat files
│   │   ├── Dreyer2019/
│   │   └── Leeuwis2021/
│   └── preprocessed/          # Preprocessed HDF5 files
│       ├── KU_mi_labram_preprocessed_trial_normalized.h5
│       └── ...
├── preprocessing/
│   ├── preprocess_KU_data.py  # KU preprocessing pipeline
│   ├── preprocess_DREYER_data.py
│   └── preprocess_NIKKI.py
├── models/
│   ├── Labram.py              # LaBraM model with LoRA
│   ├── DeepConvNet.py         # DeepConvNet implementation
│   └── __init__.py
├── engines/
│   ├── engine.py              # Supervised training engine
│   ├── test_engine.py          # Few-shot testing engine
│   ├── base_engine.py          # Base engine utilities
│   └── utils.py                # Helper functions
├── testing/
│   └── LOSO.py                 # LOSO evaluation script
├── hyperparameters/
│   ├── hyperparameters-labram-lora.yaml
│   ├── hyperparameters-deepconvnet.yaml
│   └── hyperparameters.yaml
├── results/                     # Experiment results
│   └── labram-lora/
│       └── ...
├── train.py                     # Main training entry point
├── run_loso.py                  # LOSO CLI script
├── labram-lora.job              # SLURM job file
├── subject_split.py             # Dataset splitting utilities
└── README.md                    # This file
```

---

## Configuration Files

### Hyperparameter Files

Located in `hyperparameters/`, these YAML files control:

- **Experiment settings:** model type, seed, device, epochs
- **Data settings:** dataset path, subjects, splits
- **Model hyperparameters:** learning rate, weight decay, LoRA config
- **Testing settings:** shot counts, adaptation epochs, repetitions

Example structure:
```yaml
experiment:
  model: "labram"
  seed: 111
  epochs: 1
  device: "cuda"

data:
  path: "data/preprocessed/KU_mi_labram_preprocessed_trial_normalized.h5"
  n_subjects: 54
  leave_out: null  # Set by LOSO script

labram:
  lr: 0.002
  weight_decay: 0.002
  lora: true
  test_lr: 0.00005
  test_wd: 0.05

test:
  shots: [0, 1, 2, 3, 4, 5, 10, 15, 20, 25]
  n_epochs: 20
  n_repeats: 10
```

---

## Models

### LaBraM with LoRA

- **Model:** LaBraM (Large Brain Model) for EEG
- **Adaptation:** LoRA (Low-Rank Adaptation) via PEFT
- **Input:** 4D tensor `(batch, channels, patches, patch_length)`
- **Configuration:** See `hyperparameters/hyperparameters-labram-lora.yaml`

### DeepConvNet

- **Model:** Deep Convolutional Neural Network (Braindecode-style)
- **Input:** 3D tensor `(batch, channels, time)` or 4D with patches
- **Configuration:** See `hyperparameters/hyperparameters-deepconvnet.yaml`

### Model Loading

Models are loaded in `train.py`:
- LaBraM: `load_labram()` or `load_labram_with_adapter()`
- DeepConvNet: `DeepConvNet()` class instantiation

---

## Testing and Evaluation

### TestEngine

The `TestEngine` class provides:
- Zero-shot evaluation (no adaptation)
- Few-shot evaluation (with adaptation)
- Support/query sampling
- Cross-validation for adaptation
- Result aggregation and plotting

### Evaluation Metrics

- **Accuracy:** Classification accuracy on query set
- **Loss:** Cross-entropy loss
- **Per-shot statistics:** Mean and std across repetitions

### Result Analysis

Results are saved as CSV and JSON:
- `repetition_results.csv`: Per-repetition results
- `aggregated_results.json`: Statistics across repetitions

Use `analysis/combine_results.py` to aggregate results across models and folds.

---

**Typical shapes:**
- KU (LaBraM): `(400, 62, 4, 200)` - 400 trials, 62 channels, 4 patches, 200 samples/patch
- KU (DeepConvNet): `(400, 62, 800)` - 400 trials, 62 channels, 800 time points - reshaped internally by the model.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright 2025 Urban Sirca & Lovro Brulec

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
---

