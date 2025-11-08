from collections import defaultdict
import logging
from datetime import datetime
from pathlib import Path
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy

from engines import Engine
from models import load_labram, DeepConvNet, load_labram_with_adapter, set_partial_finetune_labram
from subject_split import KUTrialDataset, SplitConfig, SplitManager
from preprocessing.preprocess_KU_data import get_ku_dataset_channels

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def get_engine(config, experiment_name = None):
    # ---------------- configs -----------------
    exp_cfg = config.get("experiment")
    data_cfg = config.get("data")

    # core exp settings
    SEED = exp_cfg["seed"]
    N_EPOCHS = exp_cfg["epochs"]
    OPTIMIZER = exp_cfg["optimizer"]
    SCHEDULER = exp_cfg["scheduler"]

    # optimizations
    NUM_WORKERS = 8
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    NON_BLOCKING = True
    USE_AMP = False

    electrodes = data_cfg.get("electrodes")
    if electrodes is None and "ku" in data_cfg.get("path").lower():
        electrodes = get_ku_dataset_channels()
        logger.info(f"USING KU CHANNELS: {electrodes}")
    else:
        raise ValueError("Please provide electrode list in config for non-KU datasets.")

    # ---------------- model ------------------
    model_name = exp_cfg["model"].lower()
    if model_name == "labram":
        model_str = "labram"
        hyperparameters = config["labram"]
        if hyperparameters["adapter_checkpoint_dir"] is None:
            model = load_labram(
                lora=hyperparameters["lora"],
                peft_config=config["peft_config"],
            )
        else:
            model = load_labram_with_adapter(
                hyperparameters["adapter_checkpoint_dir"]
            )
        if hyperparameters.get("lora") == False:
            set_partial_finetune_labram(model)
        
    elif model_name == "deepconvnet":
        hyperparameters = config.get("deepconvnet")
        model_str = "deepconvnet"
        model = DeepConvNet(
                in_chans=data_cfg.get("input_channels"),
                n_classes=data_cfg.get("num_classes"),
                input_time_length=data_cfg.get("samples"),
                final_conv_length="auto",
            )
    else:
        raise ValueError("Invalid model")



    # ---------------- run cfg ----------------
    if experiment_name is None: # for combined training we want same experiment name
        name_list = [
            model_str,
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
    TRAIN_PROP = data_cfg.get("train_proportion")
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
    train_ds = KUTrialDataset(DATASET_PATH, sm.S_train)
    val_ds = KUTrialDataset(DATASET_PATH, sm.S_val)
    test_ds = KUTrialDataset(DATASET_PATH, sm.S_test)
    train_after_stopping_ds = KUTrialDataset(DATASET_PATH, sm.S_train + sm.S_val)


    # ---------------- loaders ----------------
    TRAIN_BS = 250
    EVAL_BS = 250
    g_train = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BS,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
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
        train_after_stopping_ds, #TODO: check this !!
        batch_size=TRAIN_BS,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        generator=g_train,
        persistent_workers=PERSISTENT_WORKERS,
    )


    # ---------------- factories ------------
    def make_optimizer(model: torch.nn.Module, lr=None, wd=None):
        
        
        # use hyperparameters from config if not provided (will be provided in test engine)
        if lr is None:
            lr = float(hyperparameters.get("lr"))
        if wd is None:
            wd = float(hyperparameters.get("weight_decay"))

        # --- TEMPORARY PATCH -----------------------------------------------------
        # In the few-shot / meta-testing path, we call make_optimizer(fast)
        # where `fast` is a *list of tensors* cloned from model parameters,
        # not an nn.Module.  Normally this factory assumes a real module and
        # does model.parameters(), which fails for a plain list.
        #
        # To keep both code paths working for now, we detect this case and
        # pass the list directly to the optimizer constructor.
        # (PyTorch optimizers accept an iterable of tensors.)
        #
        # TODO(lovro): clean this up once we separate inner-loop and outer-loop
        # optimizers — the adaptation step should have its own lr/wd settings
        # instead of reusing training hyperparameters.
        # ------------------------------------------------------------------------

        if isinstance(model, list):
            if OPTIMIZER.lower() == "adamw":
                return torch.optim.AdamW(model, lr=lr, weight_decay=wd)
            elif OPTIMIZER.lower() == "adam":
                return torch.optim.Adam(model, lr=lr, weight_decay=wd)
            else:
                raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")
        else:
            if OPTIMIZER.lower() == "adamw":
                return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            elif OPTIMIZER.lower() == "adam":
                return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            else:
                raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

    def make_scheduler(optimizer: torch.optim.Optimizer):
        if SCHEDULER == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
        elif SCHEDULER == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=N_EPOCHS // exp_cfg.get("T_0"),
                T_mult=exp_cfg.get("T_mult"),
            )
        elif SCHEDULER in (None, "None"):
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {SCHEDULER}")

    # ---------------- engine -----------------
    input_channels = data_cfg.get("input_channels")
    trial_len = data_cfg.get("trial_length", data_cfg.get("samples"))
    n_patches_labram = data_cfg.get("n_patches_labram") if model_str == "labram" else None
    patch_len = data_cfg.get("patch_length") if model_str == "labram" else None
    
    
    model = model.to(DEVICE)
    samples = int(data_cfg.get("samples"))
    with torch.no_grad():
        if model_str == "labram":
            _ = model(
                x=torch.zeros(1, input_channels, n_patches_labram, patch_len, device=DEVICE),
                electrodes=electrodes,
            )
        else:
            _ = model(x=torch.zeros(1, input_channels, samples, device=DEVICE))

    return Engine(
        # --- BASE ---
        model_str=model_str,
        experiment_name=experiment_name,
        device=DEVICE,
        model=model,
        electrodes=electrodes,
        non_blocking=NON_BLOCKING,
        pin_memory=PIN_MEMORY,
        use_amp=USE_AMP,
        use_compile=False,
        use_wandb=exp_cfg.get("log_to_wandb"),
        wandb_entity="urban-sirca-vrije-universiteit-amsterdam",
        wandb_project=exp_cfg.get("wandb_project"),
        config_for_logging=config,
        save_regular_checkpoints=exp_cfg.get("save_regular_checkpoints"),
        save_regular_checkpoints_interval=exp_cfg.get("save_regular_checkpoints_interval"),
        save_best_checkpoints=exp_cfg.get("save_best_checkpoints"),
        save_final_checkpoint=exp_cfg.get("save_final_checkpoint"),
        checkpoint_dir=Path(str(exp_cfg.get("checkpoint_dir") or (Path(__file__).parent / "weights" / "checkpoints" / experiment_name))),
        # --- SPECIFIC ---
        n_epochs=N_EPOCHS,
        # data loaders
        training_set=train_loader,
        validation_set=val_loader,
        test_set=test_loader,
        train_after_stopping_set=train_after_stopping_loader,
        # loss / factories
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_factory=make_optimizer,
        scheduler_factory=make_scheduler,
        # early stopping
        early_stopping=exp_cfg.get("early_stopping"),
        early_stopping_patience=exp_cfg.get("early_stopping_patience"),
        early_stopping_delta=exp_cfg.get("early_stopping_delta"),
        # train-after-stopping
        train_after_stopping=exp_cfg.get("train_after_stopping"),
        train_after_stopping_epochs=exp_cfg.get("train_after_stopping_epochs"),
    )