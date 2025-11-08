import copy
import logging
import random
from pathlib import Path

import yaml

from train import get_engine
from .testing_utils import build_subject_list, get_checkpoint_file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_loso(config_path: str):
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
        
    test_only = base_cfg.get("lomso").get("test_only")

    # ---- subjects / folds ----------------------------------------------------
    subjects = build_subject_list(base_cfg)
    folds = subjects

    
    # ---- testing params ------------------------------------------------------
    shots_list = base_cfg.get("test").get("shots")
    n_epochs = base_cfg.get("test").get("n_epochs")
    n_repeats = base_cfg.get("test").get("n_repeats")
    models = base_cfg.get("test").get("models")

    results = {}

    if test_only:
        lomso_root = Path(base_cfg.get("lomso").get("test_root"))
        checkpoint_root = Path(base_cfg.get("lomso").get("checkpoint_root"))
        ckpt_type = base_cfg.get("lomso").get("type")
    else:
        lomso_root = Path(base_cfg.get("lomso").get("train_root"))
    subdir = base_cfg.get("lomso").get("subdir")
    if subdir:
        lomso_root = lomso_root / subdir
    lomso_root.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        logger.info(f"Running LOSO for model: {model_name}")
        
        
        for fold_idx, test_pair in enumerate(folds, start=1):

            cfg = copy.deepcopy(base_cfg)

            # set model in experiment cfg
            cfg.setdefault("experiment")["model"] = model_name

            # Configure leave_out to be the test subjects (SplitManager uses this as S_test)
            cfg.setdefault("data")["leave_out"] = [test_pair]

    
            # set experiment name so checkpoints are separated per-fold
            experiment_name = f"{model_name}_lomso_fold{fold_idx:03d}_test{test_pair}"
            
            model_folder = model_name
            if model_name == "labram" and cfg.get("labram").get("head_only_train") == True:
                model_folder += "_head_only"

            dest = lomso_root / model_folder / experiment_name
            dest.mkdir(parents=True, exist_ok=True)
            
            cfg.setdefault("experiment")["checkpoint_dir"] = str(dest)

            if test_only and model_name in ["deepconvnet", "labram"]:
                # get checkpoint file from lomso_root
                ckpt_file = get_checkpoint_file(checkpoint_root, model_name, test_pair, type=ckpt_type, head_only=cfg.get("labram").get("head_only_train"))
                if model_name == "deepconvnet":
                    cfg.setdefault("deepconvnet")["checkpoint_file"] = str(ckpt_file)
                elif model_name == "labram":
                    cfg.setdefault("labram")["adapter_checkpoint_dir"] = str(ckpt_file)
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
            use_cv_epoch_selection = base_cfg.get("lomso").get("use_cv_epoch_selection")
            cv_min_shots = base_cfg.get("lomso").get("cv_min_shots")
            _ = tester.test_all_subjects(
                shots_list=shots_list, n_epochs=n_epochs, n_repeats=n_repeats, use_cv_epoch_selection=use_cv_epoch_selection, cv_min_shots=cv_min_shots
            )

            results[f"{model_name}/{experiment_name}"] = {
                "test_pair": test_pair,
                "experiment_name": experiment_name,
                "checkpoint_dir": str(engine.checkpoint_root),
                "results_dir": str(tester.save_dir)
            }

            # finish engine and free resources
            try:
                engine.finish()
            except Exception:
                logger.exception("engine.finish() failed")

            # free memory explicitly
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