import logging
from datetime import datetime
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from engine import Engine
from models import EEGNet, load_labram
from subject_split import KUTrialDataset, SplitConfig, SplitManager
from test_fsl import test_few_shot


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

exp_cfg = config.get("experiment", {})
data_cfg = config.get("data", {})
samp_cfg = config.get("sampler", {})
opt_cfg = config.get("optimizations", {})

# core exp settings
SEED = exp_cfg["seed"]
N_EPOCHS = exp_cfg["epochs"]
OPTIMIZER = exp_cfg["optimizer"]
SCHEDULER = exp_cfg["scheduler"]

# optimizations
NUM_WORKERS = opt_cfg.get("num_workers", 0)
PIN_MEMORY = opt_cfg.get("pin_memory", False)
PERSISTENT_WORKERS = opt_cfg.get("persistent_workers", False)
NON_BLOCKING = opt_cfg.get("non_blocking", False)
USE_AMP = opt_cfg.get("use_amp", False)


# ---------------- model ------------------
model_name = exp_cfg["model"].lower()

if model_name == "labram":
    hyperparameters = config["labram"]
    model = load_labram(
        lora=hyperparameters["lora"],
        peft_config=config["peft_config"],
    )
    flatten_patches = False

elif model_name == "eegnet":
    hyperparameters = config.get("eegnet", {})
    chans = data_cfg.get("input_channels", 62)
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
        F2=hyperparameters.get(
            "F2", hyperparameters.get("F1", 8) * hyperparameters.get("D", 2)
        ),
    )
    flatten_patches = True

else:
    raise ValueError("Invalid model")

logger.info(f"HYPERPARAMETERS for {model_name}: {hyperparameters}")


# ---------------- run cfg ----------------
name_list = [
    f"{exp_cfg['model']}",
    f"lr{hyperparameters['lr']}",
    f"wd{hyperparameters['weight_decay']}",
    exp_cfg["optimizer"],
    exp_cfg["scheduler"],
    datetime.now().strftime("%H%M%S"),
]
experiment_name = "_".join(name_list)

logger.info(f"Experiment name: {experiment_name}")
logger.info(f"Experiment config: {exp_cfg}")


# ---------------- device & seeds ---------
if exp_cfg["device"] == "mps":
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device(exp_cfg["device"] if torch.cuda.is_available() else "cpu")

logger.info(f"USING DEVICE: {DEVICE}")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ---------------- data/splits ------------
DATASET_PATH = data_cfg["path"]
SUBJECT_IDS = data_cfg.get("subjects") or range(1, exp_cfg["n_subjects"] + 1)
TRAIN_PROP = data_cfg.get("train_proportion", 0.90)
LEAVE_OUT = data_cfg.get("leave_out")
M_LEAVE_OUT = data_cfg.get("m_leave_out")

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

# ---- datasets ----
train_ds = KUTrialDataset(DATASET_PATH, sm.S_train, flatten_patches=flatten_patches)
val_ds = KUTrialDataset(DATASET_PATH, sm.S_val, flatten_patches=flatten_patches)
test_ds = KUTrialDataset(DATASET_PATH, sm.S_test, flatten_patches=flatten_patches)
train_after_stopping_ds = KUTrialDataset(DATASET_PATH, sm.S_train + sm.S_val)


# ---------------- loaders ----------------
TRAIN_BS = samp_cfg.get("train_batch_size")
EVAL_BS = samp_cfg.get("eval_batch_size", TRAIN_BS)
DROP_LAST = samp_cfg.get("drop_last", False)
SHUF_SUBJ = samp_cfg.get("shuffle_subjects", True)
SHUF_TRIALS = samp_cfg.get("shuffle_trials", True)

g_train = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_BS,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=DROP_LAST,
    generator=g_train,
    persistent_workers=PERSISTENT_WORKERS,
)
val_loader = DataLoader(
    val_ds,
    batch_size=EVAL_BS,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
    persistent_workers=PERSISTENT_WORKERS,
)
test_loader = DataLoader(
    test_ds,
    batch_size=EVAL_BS,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
    persistent_workers=PERSISTENT_WORKERS,
)
train_after_stopping_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_BS,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=DROP_LAST,
    generator=g_train,
    persistent_workers=PERSISTENT_WORKERS,
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

if SCHEDULER == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
elif SCHEDULER == "CosineAnnealingWarmRestarts":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=N_EPOCHS // exp_cfg.get("T_0", 2),
        T_mult=exp_cfg.get("T_mult", 2),
    )
elif SCHEDULER in (None, "None"):
    scheduler = None
else:
    raise ValueError(f"Unsupported scheduler: {SCHEDULER}")


# ---------------- engine -----------------
model_str = exp_cfg["model"].lower()
input_channels = data_cfg.get("input_channels", 62)
trial_len = data_cfg.get("trial_length", data_cfg.get("samples", 800))
n_patches_labram = data_cfg.get("n_patches_labram") if model_str == "labram" else None
patch_len = data_cfg.get("patch_length") if model_str == "labram" else None

experiment = Engine(
    # core
    model=model,
    model_str=model_str,
    experiment_name=experiment_name,
    device=DEVICE,
    n_epochs=N_EPOCHS,

    # data
    training_set=train_loader,
    validation_set=val_loader,
    test_set=test_loader,
    train_after_stopping_set=train_after_stopping_loader,

    # optimization
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=None,  # default CE inside Engine

    # shapes
    input_channels=input_channels,
    trial_length=trial_len,
    n_patches_labram=n_patches_labram,
    patch_length=patch_len,

    # model extras
    electrodes=data_cfg.get("electrodes"),

    # perf
    use_compile=opt_cfg.get("use_compile", False),
    non_blocking=NON_BLOCKING,
    pin_memory=PIN_MEMORY,
    use_amp=USE_AMP,

    # logging
    use_wandb=exp_cfg.get("log_to_wandb", False),
    wandb_entity="urban-sirca-vrije-universiteit-amsterdam",
    wandb_project="EEG-FM",

    # checkpoints
    save_regular_checkpoints=exp_cfg.get("save_regular_checkpoints", False),
    save_final_checkpoint=exp_cfg.get("save_final_checkpoint", True),
    save_best_checkpoints=exp_cfg.get("save_best_checkpoints", True),
    save_regular_checkpoints_interval=exp_cfg.get("save_regular_checkpoints_interval", 10),
    checkpoint_dir=(Path(__file__).parent / "weights" / "checkpoints" / experiment_name),

    # early stopping
    early_stopping=exp_cfg.get("early_stopping", True),
    early_stopping_patience=exp_cfg.get("early_stopping_patience", 10),
    early_stopping_delta=exp_cfg.get("early_stopping_delta", 0.0),

    # train-after-stopping
    train_after_stopping=exp_cfg.get("train_after_stopping", False),
    train_after_stopping_epochs=exp_cfg.get("train_after_stopping_epochs", 0),

    # logging-only
    config_for_logging=config,
)


# ---------------- run --------------------
if __name__ == "__main__":
    try:
        experiment.setup_optimizations()
        experiment.train()
        experiment.test()
    finally:
        train_ds.close()
        val_ds.close()
        test_ds.close()
        experiment.finish()

    # test_few_shot(model, test_loader, n_shots=10, n_epochs=10)
