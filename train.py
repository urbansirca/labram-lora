import logging
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import wandb
import yaml

from engine import Engine
from labram import load_labram
from subject_split import KUTrialDataset, SplitConfig, SplitManager, SubjectBatchSampler
from utils import get_optimizer_scheduler

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # This makes logs appear in terminal
)
logger = logging.getLogger(__name__)


with open("hyperparameters.yaml", "r") as f:
    config = yaml.safe_load(f)

exp_cfg   = config.get("experiment", {})
data_cfg  = config.get("data", {})
samp_cfg  = config.get("sampler", {})

experiment_name = "_".join([exp_cfg["model"], datetime.now().strftime("%H%M%S")])

logger.info(f"Experiment name: {experiment_name}")
logger.info(f"Experiment config: {config['experiment']}")

if exp_cfg["model"] == "labram":
    hyperparameters = config["labram"]
    logger.info(f"HYPERPARAMETERS for labram: {hyperparameters}")
    model = load_labram(
        lora=hyperparameters["lora"],
        peft_config=config["peft_config"],
    )
else:
    raise ValueError("Invalid model")

# Experiment conf
SEED = exp_cfg["seed"]
DEVICE    = torch.device(exp_cfg["device"] if torch.cuda.is_available() else "cpu")
N_EPOCHS  = exp_cfg["epochs"]
META = exp_cfg["meta"]
OPTIMIZER = exp_cfg["optimizer"]
SCHEDULER = exp_cfg["scheduler"]

# Dataset conf
DATASET_PATH = data_cfg["path"]
SUBJECT_IDS  = data_cfg.get("subjects") or exp_cfg["shuffled_subjects"]
TRAIN_PROP   = data_cfg.get("train_proportion", 0.90)
LEAVE_OUT    = data_cfg.get("leave_out")
M_LEAVE_OUT  = data_cfg.get("m_leave_out")

# Sample conf
TRAIN_BS     = samp_cfg.get("train_batch_size", hyperparameters["batch_size"])
EVAL_BS      = samp_cfg.get("eval_batch_size", TRAIN_BS)
DROP_LAST    = samp_cfg.get("drop_last", False)
SHUF_SUBJ    = samp_cfg.get("shuffle_subjects", True)
SHUF_TRIALS  = samp_cfg.get("shuffle_trials", True)
NUM_WORKERS  = samp_cfg.get("num_workers", 0)
PIN_MEMORY   = samp_cfg.get("pin_memory", False)

# ---- build splits ----
split_cfg = SplitConfig(
    subject_ids=SUBJECT_IDS,
    m_leave_out=M_LEAVE_OUT,
    subject_ids_leave_out=LEAVE_OUT,
    train_procent=TRAIN_PROP,
    seed=SEED,
)
sm = SplitManager(split_cfg)
logger.info(f"Train subjects: {sm.S_train}")
logger.info(f"Val subjects:   {sm.S_val}")
logger.info(f"Test subjects:  {sm.S_test}")

# ---- datasets ----
train_ds = KUTrialDataset(DATASET_PATH, sm.S_train)
val_ds   = KUTrialDataset(DATASET_PATH, sm.S_val)
test_ds  = KUTrialDataset(DATASET_PATH, sm.S_test)

# --- subject-pure loaders (one subject at a time) ---
train_loader = DataLoader(
    train_ds,
    batch_sampler=SubjectBatchSampler(
        train_ds,
        batch_size=TRAIN_BS,
        shuffle_subjects=SHUF_SUBJ,
        shuffle_trials=SHUF_TRIALS,
        drop_last=DROP_LAST,
        seed=SEED,                       
        subject_order=sm.S_train,
    ),
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_ds,
    batch_sampler=SubjectBatchSampler(
        val_ds,
        batch_size=EVAL_BS,
        shuffle_subjects=False, # deterministic eval
        shuffle_trials=False,
        drop_last=False,
        seed=SEED,
        subject_order=sm.S_val,
    ),
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_ds,
    batch_sampler=SubjectBatchSampler(
        test_ds,
        batch_size=EVAL_BS,
        shuffle_subjects=False, # deterministic test
        shuffle_trials=False,
        drop_last=False,
        seed=SEED,
        subject_order=sm.S_test,
    ),
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

optimizer, scheduler = get_optimizer_scheduler(OPTIMIZER, SCHEDULER)


experiment = Engine(
    model=model,
    config=config,
    hyperparameters=hyperparameters,
    experiment_name=experiment_name,
    n_epochs=N_EPOCHS,
    device=DEVICE,
    training_set=train_loader, # DataLoader
    validation_set=val_loader, # DataLoader
    test_set=test_ds,          # DataLoader
    optimizer=optimizer,
    scheduler=scheduler
)
