import copy
import logging
from pathlib import Path

import yaml
import gc
import torch
from train import get_engine
from engines import TestEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_loso(config_path: str):
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    subjects = base_cfg.get("data").get("subjects") or range(1, base_cfg.get("experiment").get("n_subjects") + 1)

    # testing params
    shots_list = base_cfg.get("test").get("shots")
    n_epochs = base_cfg.get("test").get("n_epochs")
    n_repeats = base_cfg.get("test").get("n_repeats")
    model_name = base_cfg.get("test").get("model")
    model_folder_name = base_cfg.get("test").get("model_folder_name")


    folder = Path("results")
    loso_root = folder / model_folder_name
    loso_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running LOSO for model: {model_folder_name} on subjects: {subjects}")
    for fold_idx, subject_id in enumerate(subjects, start=1):

        cfg = copy.deepcopy(base_cfg)

        # set model in experiment cfg
        cfg.setdefault("experiment")["model"] = model_name
        cfg.setdefault("data")["leave_out"] = [subject_id]

        # set experiment name so checkpoints are separated per-fold
        experiment_name = f"{model_folder_name}_fold{fold_idx:03d}_subj{subject_id}"
        dest = loso_root / experiment_name
        dest.mkdir(parents=True, exist_ok=True)
        
        cfg.setdefault("experiment")["checkpoint_dir"] = str(dest)

        # create engine and tester
        engine = get_engine(cfg, experiment_name=experiment_name)
        engine.checkpoint_root = dest
        tester = TestEngine(
            engine=engine,
            test_ds=engine.test_set.dataset,
            run_size=100,
            save_dir= dest / Path('test_results'),
            test_lr=cfg.get(model_name).get("test_lr"),
            test_wd=cfg.get(model_name).get("test_wd"),
        ) 

        logger.info(f"Starting fold {fold_idx}/{len(subjects)} for model {model_folder_name}: test={subject_id} experiment_name={experiment_name}")
        logger.info(f"Checkpoints and results will be saved to {str(engine.checkpoint_root)}")

        # Train
        engine.train()

        # Evaluate across requested shot counts
        tester.test_all_subjects(
            shots_list=shots_list,
            n_epochs=n_epochs,
            n_repeats=n_repeats,
        )

        # Cleanup
        engine.finish()
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()