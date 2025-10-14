# LaBraM LoRA — Project Overview

Extensible codebase for zero/few-shot EEG motor imagery classification with:
- Supervised training (DeepConvNet, EEGNet, LaBraM + LoRA)
- Meta-learning and few-shot evaluation across subjects (LOMSO)
- Reproducible dataset preprocessing pipelines (KU, Dreyer/ABC, Nikki)
- Robust testing utilities (few-shot with support/query and CV-based epoch selection)

This document describes the purpose of each Python module and how the pieces fit together.

---

## Quickstart

- Configure training/evaluation in:
  - [hyperparameters/hyperparameters.yaml](hyperparameters/hyperparameters.yaml)
  - [hyperparameters/meta_hyperparameters.yaml](hyperparameters/meta_hyperparameters.yaml)
- Preprocess a dataset (KU example):
  - See [`preprocessing/preprocess_KU_data.preprocess_ku_data`](preprocessing/preprocess_KU_data.py)
- Run LOMSO experiments:
  - Supervised: `python run_lomso.py`
  - Meta-learning: `python run_lomso_meta.py`
- Single-run training: `python train.py`

Example commands:
```bash
# Supervised LOMSO runs (fold orchestration)
python run_lomso.py

# Meta-learning LOMSO runs (alternating scheduler)
python run_lomso_meta.py

# Single-run supervised (uses hyperparameters.yaml)
python train.py
```

Environment examples:
```bash
# GPU
conda env create -f enviroments/environment-gpu.yml
conda activate eeg2

# CPU-only
conda env create -f enviroments/environment-cpu.yml

```

---

## Data format (HDF5)

Datasets are stored as:
- Root groups per subject: /s{sid}
- Datasets per subject:
  - X: trials × channels × [patches] × time
  - Y: trials (int, class labels)

Typical shapes:
- KU (LaBraM-ready): X ≈ (T, 62, 4, 200), Y ≈ (T,)
- Dreyer/ABC (after resample + patching): X ≈ (T, 27, 5, 200), Y ≈ (T,)
- Nikki: X ≈ (T, 16, time), Y ≈ (T,) — can be patched later if needed

A quick inspector is provided in [preprocessing/test_datasets.py](preprocessing/test_datasets.py).

---

## Core training engines

- [`engine.Engine`](engine.py)
  - Supervised training loop (epochs, early stopping, checkpointing, W&B logging).
  - Accepts factories for optimizer/scheduler and dataloaders.
  - Writes final metrics to JSON on finish.
- [`base_engine.BaseEngine`](base_engine.py)
  - Common runtime utilities (metric logging, elapsed-time tracking, W&B log funnel).
- [`meta_engine.MetaEngine`](meta_engine.py)
  - Meta-learning/episodic training loop (meta-iterations, validation cadence).
  - Controls meta-train/val/test datasets and episodic sampling.

Key methods to look at:
- [`engine.Engine`](engine.py) construction and `finish()`
- [`base_engine.BaseEngine.log_metrics`](base_engine.py)
- [`meta_engine.MetaEngine.finish`](meta_engine.py)

---

## High-level runners

- [`run_lomso.run_lomso`](run_lomso.py)
  - Builds pairwise subject folds for LOMSO supervised experiments.
  - Creates per-fold experiment directories and delegates to [`train.get_engine`](train.py).
  - Supports test-only mode (load precomputed checkpoints).
- [`run_lomso_meta.run_lomso`](run_lomso_meta.py)
  - Same concept for meta-learning experiments using [`meta_train.get_meta_engine`](meta_train.py).
  - Includes alternating training mode (e.g., train_alternating), then few-shot evaluation.

Utilities:
- Fold discovery, seeding, and per-model checkpoint routing
- Best/final checkpoint selection helpers (`get_checkpoint_file`)

---

## Entry points

- [`train.get_engine`](train.py)
  - Parses config, instantiates model, datasets, loaders, and the [`engine.Engine`](engine.py).
  - Warm-graphs the model with a dummy input (accounts for LaBraM patches or standard 3D inputs).
  - Returns `(engine, tester)` if requested.

- [`meta_train.get_meta_engine`](meta_train.py)
  - Similar to `get_engine` but for meta-learning.
  - Builds the [`meta_engine.MetaEngine`](meta_engine.py) and a tester.

---

## Models

- [`models/DeepConvNet.DeepConvNet`](models/DeepConvNet.py)
  - Local reimplementation aligned with Braindecode’s DeepConvNet.
  - Accepts (B, C, T) or (B, C, P, T) inputs. If 4D, it flattens P×T to time.
  - Helper: [`models.DeepConvNet.freeze_all_but_head_deepconvnet`](models/DeepConvNet.py)
- [`models/Labram.load_labram`](models/Labram.py)
  - Loads LaBraM with optional PEFT-LoRA adapters via `peft`.
  - Adapter loading: [`models.Labram.load_labram_with_adapter`](models/Labram.py)
  - Helper: [`models.Labram.freeze_all_but_head_labram`](models/Labram.py)
- [`models/mirepnet_model`](models/mirepnet_model.py)
  - MIREPNet variant with masking pretrain pathway (optional LoRA for adapters).

Model registry:
- [models/__init__.py](models/__init__.py)

---

## Testing and few-shot utilities

- [`test_engine.TestEngine`](test_engine.py)
  - Few-shot testing driver that:
    - Samples support/query per subject
    - Fine-tunes either the head-only or full model (configurable)
    - Repeats across multiple shots and random seeds; aggregates to JSON and plots
  - Includes cross-validation epoch selection for the adaptation step.

- Key helpers from [`meta_helpers`](meta_helpers.py):
  - Episode indexing: [`meta_helpers.build_episode_index`](meta_helpers.py)
  - Support sampling: `sample_support`
  - Query sampling: `sample_query`
  - Data fetch: [`meta_helpers.fetch_by_indices`](meta_helpers.py)

---

## Datasets and splits

- [`subject_split.KUTrialDataset`](subject_split.py)
  - Fast HDF5-backed dataset.
  - Emits `(X, Y, sid)` with optional channel subset for MIREPNet-45.
- [`subject_split.SplitManager`](subject_split.py)
  - Builds train/val/test subject splits:
    - Leave-M-out or explicit `leave_out`
    - Train/validation proportion controlled via config

Channel lists:
- KU 62-channel map: [`preprocessing/preprocess_KU_data.get_ku_dataset_channels`](preprocessing/preprocess_KU_data.py)
- Dreyer/ABC 27-channel map: see [`preprocessing/preprocess_ABC_data.EEG_27`](preprocessing/preprocess_ABC_data.py)

---

## Preprocessing pipelines

- Dreyer/ABC:
  - [`preprocessing/preprocess_ABC_data.convert_subject`](preprocessing/preprocess_ABC_data.py)
    - Robust event parsing, preprocessing (band-pass, notch, CAR, optional ICA)
    - Epoch extraction: [`preprocessing/preprocess_ABC_data.epochs_from_events`](preprocessing/preprocess_ABC_data.py)
    - Resampling to target length and non-overlapping patching
  - Write H5: [`preprocessing/preprocess_ABC_data.write_h5`](preprocessing/preprocess_ABC_data.py)

- KU:
  - [`preprocessing/preprocess_KU_data.preprocess_ku_data`](preprocessing/preprocess_KU_data.py)
    - Session concatenation, leakage-safe normalization (support/query split-aware)
    - Saves per-subject stats under `norm/` for reproducibility

- Nikki:
  - [`preprocessing/preprocess_NIKKI.csvs_to_h5`](preprocessing/preprocess_NIKKI.py)
    - CSVs to HDF5 converter with per-trial stacking

- Sanity checks:
  - [preprocessing/test_datasets.py](preprocessing/test_datasets.py)
  - [preprocessing/Dreyer_moabb.py](preprocessing/Dreyer_moabb.py)

---

## Hyperparameters and studies

- Supervised:
  - [hyperparameters/hyperparameters.yaml](hyperparameters/hyperparameters.yaml)
- Meta-learning:
  - [hyperparameters/meta_hyperparameters.yaml](hyperparameters/meta_hyperparameters.yaml)
- Optuna tuning:
  - Study driver: [`hypertune_meta.main`](hypertune_meta.py)
  - Config: [hyperparameters/hyperparam_meta_tuning.yaml](hyperparameters/hyperparam_meta_tuning.yaml)

---

## Result analysis and jobs

- Result aggregation/plotting:
  - See plotting inside [`test_engine.TestEngine.plot_aggregated_results`](test_engine.py)
  - Utility scaffolding in [analyze_results.py](analyze_results.py) (if used)

- Batch jobs:
  - [labram-lora.job](labram-lora.job), [labram-lora2.job](labram-lora2.job)
  - Tip: keep secrets (e.g., W&B keys) out of version control.

---

## File-by-file summary

Top level:
- [train.py](train.py): Builds and runs [`engine.Engine`](engine.py) via [`train.get_engine`](train.py).
- [meta_train.py](meta_train.py): Builds and runs [`meta_engine.MetaEngine`](meta_engine.py) via [`meta_train.get_meta_engine`](meta_train.py).
- [engine.py](engine.py): Supervised engine and metric persistence.
- [base_engine.py](base_engine.py): Shared engine utilities (logging, W&B).
- [meta_engine.py](meta_engine.py): Episodic/meta-training engine.
- [run_lomso.py](run_lomso.py): Supervised LOMSO folds orchestration.
- [run_lomso_meta.py](run_lomso_meta.py): Meta-learning LOMSO orchestration.
- [test_engine.py](test_engine.py): Few-shot testing across subjects, JSON outputs, plots.
- [test_ng_models.py](test_ng_models.py): Batch-evaluates a directory of pre-trained models on selected subjects.
- [subject_split.py](subject_split.py): Dataset and split management.
- [meta_helpers.py](meta_helpers.py): Episodic indexing and sampling helpers.
- [hypertune_meta.py](hypertune_meta.py): Optuna hyperparameter optimization (meta).
- [setup.sh](setup.sh): Environment bootstrap (if used).
- [analyze_results.py](analyze_results.py): Result-analysis scaffold (optional).

Models:
- [models/DeepConvNet.py](models/DeepConvNet.py): DeepConvNet implementation and freeze helpers.
- [models/Labram.py](models/Labram.py): LaBraM loader with PEFT-LoRA support and adapter loading.
- [models/mirepnet_model.py](models/mirepnet_model.py): MIREPNet model and pretraining utilities.
- [models/__init__.py](models/__init__.py): Model registry.

Preprocessing:
- [preprocessing/preprocess_KU_data.py](preprocessing/preprocess_KU_data.py): KU pipeline and channel utilities.
- [preprocessing/preprocess_ABC_data.py](preprocessing/preprocess_ABC_data.py): Dreyer/ABC pipeline (events, ICA, patching).
- [preprocessing/preprocess_NIKKI.py](preprocessing/preprocess_NIKKI.py): Nikki dataset CSV-to-H5.
- [preprocessing/test_datasets.py](preprocessing/test_datasets.py): Inspect existing HDF5s.
- [preprocessing/Dreyer_moabb.py](preprocessing/Dreyer_moabb.py): Minimal MOABB example.

Configs:
- [hyperparameters/hyperparameters.yaml](hyperparameters/hyperparameters.yaml): Main supervised config.
- [hyperparameters/meta_hyperparameters.yaml](hyperparameters/meta_hyperparameters.yaml): Meta-learning config.
- [hyperparameters/hyperparam_meta_tuning.yaml](hyperparameters/hyperparam_meta_tuning.yaml): Optuna config.

Environments:
- [enviroments/environment-gpu.yml](enviroments/environment-gpu.yml)
- [enviroments/environment-cpu.yml](enviroments/environment-cpu.yml)
- [enviroments/requirements.txt](enviroments/requirements.txt)

---

## Tips

- LaBraM inputs must be 4D: (B, C, P, T). Configure:
  - data.n_patches_labram, data.patch_length, data.trial_length
- DeepConvNet accepts (B, C, T) or (B, C, P, T) and flattens patches internally.
- To head-only fine-tune during few-shot testing, use:
  - `labram.head_only_test` / `deepconvnet.head_only_test` in [hyperparameters/hyperparameters.yaml](hyperparameters/hyperparameters.yaml)
- Use `test_only: true` and `checkpoint_dir` to re-evaluate saved models without retraining.

For any symbol or file above, see the linked source for details.