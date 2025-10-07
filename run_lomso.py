import argparse
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


def run_lomso(
    config_path: str,
    max_folds: int = None,
):
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

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
    models = base_cfg.get("test", {}).get("models", ["deepconvnet", "labram"])

    results = {}

    lomso_root = Path("lomso")
    # if lomso_root.exists():
    #     raise ValueError(f"Directory {lomso_root} already exists. Please move or delete it before running.")
    lomso_root.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        logger.info(f"Running LOMSO for model: {model_name}")

        for fold_idx, test_pair in enumerate(folds, start=1):
            cfg = copy.deepcopy(base_cfg)

            # set model in experiment cfg
            cfg.setdefault("experiment", {})["model"] = model_name

            # Configure leave_out to be the test subjects (SplitManager uses this as S_test)
            cfg.setdefault("data", {})["leave_out"] = list(test_pair)

            # set experiment name so checkpoints are separated per-fold

            experiment_name = f"{model_name}_lomso_fold{fold_idx:03d}_test{'-'.join(map(str,test_pair))}"

            dest = lomso_root / model_name / experiment_name
            dest.mkdir(parents=True, exist_ok=True)

            # if any(dest.iterdir()):
            #     raise ValueError(f"Directory {dest} already exists and is not empty. Please move or delete it before running.")

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
    run_lomso("hyperparameters/hyperparameters.yaml") # set max_folds to limit number of folds for development purposes
