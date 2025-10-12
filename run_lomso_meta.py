import copy
import itertools
import random
import logging
import shutil
from pathlib import Path

import yaml

from train import get_engine


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_subject_list(config):
    data_cfg = config.get("data", {})
    exp_cfg = config.get("experiment", {})
    subjects = data_cfg.get("subjects")
    if subjects is None:
        n = exp_cfg.get("n_subjects")
        if n is None:
            raise ValueError(
                "No subjects list in config and experiment.n_subjects not set"
            )
        subjects = list(range(1, n + 1))
    return list(subjects)


def get_checkpoint_file(root_dir, model_name, leave_out_subjs, type="best"):
    """
    gets the checkpoint file for a given model and leave_out subjects for LOMSO. 
    type can be best or final
    """
    
    model_dir = Path(root_dir) / model_name
    
    print(f"Looking for model directory {model_dir}")
    if not model_dir.exists():
        raise ValueError(f"Model directory {model_dir} does not exist")
    # find the experiment dir with the matching leave_out subjects
    for experiment_dir in model_dir.iterdir():
        if not experiment_dir.is_dir():
            continue
        # check if leave_out matches
        if f"test{'-'.join(map(str, sorted(leave_out_subjs)))}" in experiment_dir.name:
            # found the matching experiment dir
            if model_name == "deepconvnet":
                if type == "best":
                    ckpt_file = experiment_dir / "best_val_checkpoint.pt"
                elif type == "final":
                    # find .pt file that starts with final_checkpoint (there should be only one)
                    final_ckpt_files = list(experiment_dir.glob("final_checkpoint*.pt"))
                    if not final_ckpt_files:
                        raise ValueError(f"No final_checkpoint*.pt file found in {experiment_dir}")
                    if len(final_ckpt_files) > 1:
                        raise ValueError(f"Multiple final_checkpoint*.pt files found in {experiment_dir}")
                    ckpt_file = final_ckpt_files[0]
            elif model_name == "labram":
                if type == "best":
                    ckpt_file = experiment_dir / "best_val_checkpoint" # its a lora directory
                elif type == "final":
                    # find the subfolder that starts with final_checkpoint (there should be only one)
                    final_ckpt_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("final_checkpoint")]
                    if not final_ckpt_dirs:
                        raise ValueError(f"No final_checkpoint* directory found in {experiment_dir}")
                    if len(final_ckpt_dirs) > 1:
                        raise ValueError(f"Multiple final_checkpoint* directories found in {experiment_dir}")
                    ckpt_file = final_ckpt_dirs[0]
            else:
                raise ValueError(f"Unknown model name {model_name}")
    print(f"Found checkpoint file {ckpt_file} for model {model_name} leave_out {leave_out_subjs} type {type}")
    return ckpt_file


def run_lomso(config_path: str):
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
        
    max_folds = base_cfg.get("lomso").get("max_folds")
    test_only = base_cfg.get("lomso").get("test_only")
    skip_models = base_cfg.get("lomso").get("skip_models", [])

    subjects = build_subject_list(base_cfg)
    N = len(subjects)
    if N < 4:
        raise ValueError("Need at least 4 subjects for 2-test/2-val LOMSO")

    # Build folds so that each subject is in test exactly once and in val exactly once.
    # Approach: shuffle subjects deterministically using config seed (if present),
    # pair adjacent subjects into M pairs, then for i in 0..M-1 create fold with
    # test = pair[i], val = pair[(i+1) mod M]. This yields exactly M folds and each
    # subject appears in test once and val once.
    exp_seed = base_cfg.get("experiment", {}).get("seed")
    rng = random.Random(exp_seed) if exp_seed is not None else random.Random()

    if len(subjects) % 2 != 0:
        raise ValueError("Number of subjects must be even for pairwise LOMSO folding")

    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    pairs = [tuple(shuffled[i : i + 2]) for i in range(0, len(shuffled), 2)]

    # For LOMSO we only specify test pairs here; validation will be handled
    # by the SplitManager inside get_engine using train_proportion from config.
    folds = [tuple(sorted(p)) for p in pairs]
    if max_folds is not None:
        folds = folds[:max_folds]

    logger.info(f"Built {len(folds)} test folds (pairwise), max_folds={max_folds}")
    logger.info(f"Test pairs: {folds}")

    shots_list = base_cfg.get("test", {}).get("shots", [0, 1, 2, 5, 10, 15, 20, 25])
    n_epochs = base_cfg.get("test", {}).get("n_epochs", 10)
    n_repeats = base_cfg.get("test", {}).get("n_repeats", 10)
    models = base_cfg.get("test", {}).get("models")

    results = {}

    if not test_only:
        lomso_root = Path("lomso")
    else:
        lomso_root = Path("test_only_lomso")
        checkpoint_root = Path("lomso/lomso_run1") # I renamed it from lomso --> lomso_supervised to avoid confusion
    # if lomso_root.exists():
    #     raise ValueError(f"Directory {lomso_root} already exists. Please move or delete it before running.")
    lomso_root.mkdir(parents=True, exist_ok=True)

    subdir = "run1"
    lomso_root = lomso_root / subdir
    lomso_root.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        logger.info(f"Running LOMSO for model: {model_name}")
        # skip deepconvnet
        if model_name in skip_models:
            logger.info(f"Skipping {model_name} for now")
            continue
        
        
        for fold_idx, test_pair in enumerate(folds, start=1):
            # if fold_idx in [1,2,3]:
            #     logger.info(f"Skipping fold {fold_idx} for development purposes")
            #     continue

            cfg = copy.deepcopy(base_cfg)
            
            # just for now
            # cfg.setdefault(model_name, {})["head_only_test"] = head_only

            # set model in experiment cfg
            cfg.setdefault("experiment", {})["model"] = model_name
            
            if model_name == "mirepnet":
                cfg.setdefault("data", {})["path"] = "data/preprocessed/KU_mi_labram_preprocessed_trial_norm_NG_format.h5"

            # Configure leave_out to be the test subjects (SplitManager uses this as S_test)
            cfg.setdefault("data", {})["leave_out"] = list(test_pair)
    
            # set experiment name so checkpoints are separated per-fold
            experiment_name = f"{model_name}_lomso_fold{fold_idx:03d}_test{'-'.join(map(str,test_pair))}"
            
            dest = lomso_root / model_name / experiment_name
            dest.mkdir(parents=True, exist_ok=True)
            
            cfg.setdefault("experiment", {})["checkpoint_dir"] = str(dest)

            # if any(dest.iterdir()):
            #     raise ValueError(f"Directory {dest} already exists and is not empty. Please move or delete it before running.")

            if test_only and model_name in ["deepconvnet", "labram"]:
                # get checkpoint file from lomso_root
                ckpt_file = get_checkpoint_file(checkpoint_root, model_name, test_pair, type="best")
                if model_name == "deepconvnet":
                    cfg.setdefault("deepconvnet", {})["checkpoint_file"] = str(ckpt_file)
                elif model_name == "labram":
                    cfg.setdefault("labram", {})["adapter_checkpoint_dir"] = str(ckpt_file)
                else:
                    raise ValueError(f"Unknown model name {model_name}")
                                
            # create engine and tester
            engine, tester = get_engine(
                cfg, with_tester=True, experiment_name=experiment_name
            )

            # Make engine and tester use the lomso folder as its checkpoint root
            engine.checkpoint_root = dest
            tester.save_dir = dest

            logger.info(
                f"Starting fold {fold_idx}/{len(folds)} for model {model_name}: test={test_pair} experiment_name={experiment_name}"
            )
            logger.info(f"Checkpoints and results will be saved to {str(engine.checkpoint_root)}")

            # train
            if not test_only:
                engine.train()

            # run tests (tester.test_all_subjects uses the split's test subjects)
            all_results = tester.test_all_subjects(
                shots_list=shots_list, n_epochs=n_epochs, n_repeats=n_repeats
            )

            results[f"{model_name}/{experiment_name}"] = {
                "test_pair": test_pair,
                "experiment_name": experiment_name,
                "checkpoint_dir": str(engine.checkpoint_root),
                "results_dir": str(tester.save_dir)
            }

            # finish engine (closes wandb run, writes final_metrics) and free resources
            try:
                engine.finish()
            except Exception:
                logger.exception("engine.finish() failed")

            # free memory explicitly (helps when running many folds sequentially)
            try:
                import gc
                import torch as _torch

                del engine
                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass

        outp = lomso_root / "lomso.yaml"
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            yaml.safe_dump(results, f)


if __name__ == "__main__":
    run_lomso("hyperparameters/meta_hyperparameters.yaml") # set max_folds to limit number of folds for development purposes
    
    
