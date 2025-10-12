import copy
import itertools
import random
import logging
import shutil
from pathlib import Path
import os

import yaml

from train import get_engine

# subjects list (model_f0 -> subject 35)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

subjs = [
    35,
    47,
    46,
    37,
    13,
    27,
    12,
    32,
    53,
    54,
    4,
    40,
    19,
    41,
    18,
    42,
    34,
    7,
    49,
    9,
    5,
    48,
    29,
    15,
    21,
    17,
    31,
    45,
    1,
    38,
    51,
    8,
    11,
    16,
    28,
    44,
    24,
    52,
    3,
    26,
    39,
    50,
    6,
    23,
    2,
    14,
    25,
    20,
    10,
    33,
    22,
    43,
    36,
    30,
]

WEIGHTS_DIR = "weights/pretrained-models/ng_deepconvnet_pretrained"



def run(config_path: str):
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
        
    shots_list = base_cfg.get("test", {}).get("shots", [0, 1, 2, 5, 10, 15, 20, 25])
    n_epochs = base_cfg.get("test", {}).get("n_epochs", 10)
    n_repeats = base_cfg.get("test", {}).get("n_repeats", 10)
    models = base_cfg.get("test", {}).get("models", ["deepconvnet", "labram"])

    results = {}

    lomso_root = Path("ng_testing") / "raw_dataset"
    lomso_root.mkdir(parents=True, exist_ok=True)
    
    weights_path = Path(WEIGHTS_DIR)
    model_files = sorted(
        weights_path.glob("model_f*.pt"),
        key=lambda p: int(p.stem.split("model_f")[-1]))
    
    model_names = [p.stem for p in model_files]

    for model_path in model_files:
        model_path = str(model_path)
        idx = int(model_path.split("model_f")[-1].split(".")[0])
        subject = subjs[idx]
        logger.info(f"Model {model_path} corresponds to subject {subject}")

        model_name = model_path.split("/")[-1].split(".")[0]


        cfg = copy.deepcopy(base_cfg)
        
        cfg.setdefault("data", {})["path"] = "data/preprocessed/ng/KU_mi_smt.h5"
        # just for now
        # cfg.setdefault(model_name, {})["head_only_test"] = head_only

        # set model in experiment cfg
        cfg.setdefault("experiment", {})["model"] = "deepconvnet"
        cfg.setdefault("deepconvnet", {})["checkpoint_file"] = str(model_path)

        # Configure leave_out to be the test subjects (SplitManager uses this as S_test)
        cfg.setdefault("data", {})["leave_out"] = [subject]

        # set experiment name so checkpoints are separated per-fold
        experiment_name = f"test_subject_{subject}_{model_name}"
        
        dest = lomso_root / experiment_name
        dest.mkdir(parents=True, exist_ok=True)
        
        cfg.setdefault("experiment", {})["checkpoint_dir"] = str(dest)
                            
        # create engine and tester
        engine, tester = get_engine(
            cfg, with_tester=True, experiment_name=experiment_name
        )

        # Make engine and tester use the lomso folder as its checkpoint root
        engine.checkpoint_root = dest
        tester.save_dir = dest

        logger.info(
            f"Using checkpoint file from {WEIGHTS_DIR}/{model_name}/, subject {subject}"
        )
        logger.info(f"Checkpoints and results will be saved to {str(engine.checkpoint_root)}")

        # run tests (tester.test_all_subjects uses the split's test subjects)
        all_results = tester.test_all_subjects(
            shots_list=shots_list, n_epochs=n_epochs, n_repeats=n_repeats
        )

        results[f"{model_name}/{experiment_name}"] = {
            "test_subject": subject,
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
    run("hyperparameters/hyperparameters.yaml", ) # set max_folds to limit number of folds for development purposes
