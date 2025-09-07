import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import yaml

from engine import Engine
from models import EEGNet, load_labram
from subject_split import KUTrialDataset, SplitConfig, SplitManager, SubjectBatchSampler


# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------- config -----------------
with open("hyperparameters.yaml", "r") as f:
    config = yaml.safe_load(f)

exp_cfg  = config.get("experiment", {})
data_cfg = config.get("data", {})
samp_cfg = config.get("sampler", {})

experiment_name = f"{exp_cfg['model']}_{datetime.now().strftime('%H%M%S')}"
logger.info(f"Experiment name: {experiment_name}")
logger.info(f"Experiment config: {config['experiment']}")


# ---------------- model ------------------
model_name = exp_cfg["model"].lower()
if model_name == "labram":
    hyperparameters = config["labram"]
    logger.info(f"HYPERPARAMETERS for labram: {hyperparameters}")
    model = load_labram(
        lora=hyperparameters["lora"],
        peft_config=config["peft_config"],
    )
elif model_name == "eegnet":
    hyperparameters = config.get("eegnet", {})
    chans   = data_cfg.get("input_channels", 62)
    samples = data_cfg.get("samples", 1000)
    classes = data_cfg.get("num_classes", 2)
    model = EEGNet(
        nb_classes=classes,
        Chans=chans,
        Samples=samples,
        dropoutRate=hyperparameters.get("dropoutRate", 0.5),
        kernLength=hyperparameters.get("kernLength", 64),
        F1=hyperparameters.get("F1", 8),
        D=hyperparameters.get("D", 2),
        F2=hyperparameters.get("F2", hyperparameters.get("F1", 8) * hyperparameters.get("D", 2)),
    )
else:
    raise ValueError("Invalid model")


# ---------------- run cfg ----------------
SEED = exp_cfg["seed"]
DEVICE    = torch.device(exp_cfg["device"] if torch.cuda.is_available() else "cpu")
N_EPOCHS  = exp_cfg["epochs"]
META = exp_cfg["meta"]
OPTIMIZER = exp_cfg["optimizer"]
SCHEDULER = exp_cfg["scheduler"]

# set global seeds
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ---------------- data/splits ------------
DATASET_PATH = data_cfg["path"]
SUBJECT_IDS  = data_cfg.get("subjects") or exp_cfg["shuffled_subjects"]
TRAIN_PROP   = data_cfg.get("train_proportion", 0.90)
LEAVE_OUT    = data_cfg.get("leave_out")
M_LEAVE_OUT  = data_cfg.get("m_leave_out")

split_cfg = SplitConfig(
    subject_ids=SUBJECT_IDS,
    m_leave_out=M_LEAVE_OUT,
    subject_ids_leave_out=LEAVE_OUT,
    train_proportion=TRAIN_PROP,
    seed=SEED,
)
sm = SplitManager(split_cfg)
logger.info(f"Train subjects: {sm.S_train}")
logger.info(f"Val subjects:   {sm.S_val}")
logger.info(f"Test subjects:  {sm.S_test}")

train_ds = KUTrialDataset(DATASET_PATH, sm.S_train)
val_ds   = KUTrialDataset(DATASET_PATH, sm.S_val)
test_ds  = KUTrialDataset(DATASET_PATH, sm.S_test)


# ---------------- loaders ----------------
TRAIN_BS     = samp_cfg.get("train_batch_size")
EVAL_BS      = samp_cfg.get("eval_batch_size", TRAIN_BS)
DROP_LAST    = samp_cfg.get("drop_last", False)
SHUF_SUBJ    = samp_cfg.get("shuffle_subjects", True)
SHUF_TRIALS  = samp_cfg.get("shuffle_trials", True)
NUM_WORKERS  = samp_cfg.get("num_workers", 0)
PIN_MEMORY   = samp_cfg.get("pin_memory", False)

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


# ---------------- optim/sched ------------
lr = float(hyperparameters.get("lr", 1e-3))
wd = float(hyperparameters.get("weight_decay", 0.0))

if OPTIMIZER.lower() == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
elif OPTIMIZER.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
else:
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

scheduler = None
if SCHEDULER == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
elif SCHEDULER in (None, "None"):
    scheduler = None
else:
    raise ValueError(f"Unsupported scheduler: {SCHEDULER}")


# ---------------- engine -----------------
experiment = Engine(
    model=model,
    config=config,
    hyperparameters=hyperparameters,
    experiment_name=experiment_name,
    n_epochs=N_EPOCHS,
    device=DEVICE,
    training_set=train_loader, # DataLoader
    validation_set=val_loader, # DataLoader
    test_set=test_loader,      # DataLoader
    optimizer=optimizer,
    scheduler=scheduler,
    use_wandb=exp_cfg.get("log_to_wandb", False),
)


if __name__ == "__main__":
    try:
        experiment.train()
        experiment.test()
    finally:
        train_ds.close()
        val_ds.close()
        test_ds.close()
        experiment.finish()