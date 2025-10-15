import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml
import torch
import torch.nn as nn

from engines import MetaEngine
from models import EEGNet, load_labram, load_labram_with_adapter, DeepConvNet
from subject_split import KUTrialDataset, SplitConfig, SplitManager
from preprocessing.preprocess_KU_data import get_ku_dataset_channels
from engines import TestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def get_meta_engine(config, with_tester = False, experiment_name = None, model= None, model_str= None, model_hyperparameters = None):
    # config
    exp_cfg = config.get("experiment")
    data_cfg = config.get("data")
    meta_cfg = config.get("meta")
    opt_cfg = config.get("optimizations")

    # core exp settings
    SEED = int(exp_cfg.get("seed"))
    META_ITERS = int(exp_cfg["meta_iterations"])
    OPTIMIZER = exp_cfg["optimizer"]
    SCHEDULER = exp_cfg["scheduler"]
    SUP_OPT = OPTIMIZER
    SUP_SCHED = SCHEDULER

    # perf
    USE_AMP = bool(opt_cfg.get("use_amp"))
    USE_COMPILE = bool(opt_cfg.get("use_compile"))
    NON_BLOCKING = bool(opt_cfg.get("non_blocking"))
    PIN_MEMORY = bool(opt_cfg.get("pin_memory"))

    # -------- model ------------
    model_name = exp_cfg["model"].lower()
    if model_name == "labram":
        model_str = "labram"
        hyperparameters = config.get("labram")
        if hyperparameters.get("adapter_checkpoint_dir") is None:
            model = load_labram(
                    lora=hyperparameters.get("lora"),
                    peft_config=config.get("peft_config"),
            )
        else:
            model = load_labram_with_adapter(
                hyperparameters.get("adapter_checkpoint_dir")
            )
        
    elif model_name == "deepconvnet":
        model_str = "deepconvnet"
        hyperparameters = config.get("deepconvnet")
        model = DeepConvNet(
            in_chans=int(data_cfg.get("n_channels")),
            n_classes=int(data_cfg.get("num_classes")),
            input_time_length=int(data_cfg.get("samples")),
            final_conv_length="auto",
        )
        model_str = "deepconvnet"
    elif model_name == "eegnet":
        model_str = "eegnet"
        hyperparameters = config.get("eegnet")
        model = EEGNet(
            nb_classes=int(data_cfg.get("num_classes")),
            Chans=int(data_cfg.get("n_channels")),
            Samples=int(data_cfg.get("samples")),
            dropoutRate=hyperparameters.get("dropoutRate"),
            kernLength=hyperparameters.get("kernLength"),
            F1=hyperparameters.get("F1"),
            D=hyperparameters.get("D"),
            F2=hyperparameters.get("F2"),
        )
        model_str = "eegnet"
    else:
        raise ValueError("experiment.model must be one of: labram, eegnet")

    # ---------------- run cfg ----------------
    if experiment_name is None: # for combined training we want same experiment name
        name_list = [
            model_str,
            exp_cfg["optimizer"],
            exp_cfg["scheduler"],
            datetime.now().strftime("%H%M%S"),
        ]
        experiment_name = "_".join(name_list)

    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Experiment config: {exp_cfg}")

    # -------- device & seeds ---
    if exp_cfg.get("device") == "mps":
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device(
            exp_cfg.get("device") if torch.cuda.is_available() else "cpu"
        )
    logger.info(f"USING DEVICE: {DEVICE}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # -------- data/splits ------
    DATASET_PATH = data_cfg["path"]
    SUBJECT_IDS = data_cfg.get("subjects") or range(1, exp_cfg.get("n_subjects") + 1)
    TRAIN_PROP = float(data_cfg.get("train_proportion"))
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

    # NOTE: MetaEngine expects raw trials; per-model shaping happens inside fetch_by_indices.
    train_ds = KUTrialDataset(DATASET_PATH, sm.S_train)
    val_ds = KUTrialDataset(DATASET_PATH, sm.S_val)
    test_ds = KUTrialDataset(DATASET_PATH, sm.S_test)

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

    def make_scheduler(opt: torch.optim.Optimizer):
        if SCHEDULER == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=META_ITERS, eta_min=float(exp_cfg.get("eta_min")))
        elif SCHEDULER == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=int(exp_cfg.get("T_0")),
                T_mult=int(exp_cfg.get("T_mult")),
                eta_min=float(exp_cfg.get("eta_min")),
            )
        elif SCHEDULER in (None, "None"):
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {SCHEDULER}")
    
    def make_sup_optimizer(params: Iterable[torch.nn.Parameter]):
        opt = SUP_OPT.lower()
        if opt == "adamw":
            return torch.optim.AdamW(list(params), lr=float(hyperparameters.get("sup_lr")), weight_decay=float(hyperparameters.get("sup_wd")))
        elif opt == "adam":
            return torch.optim.Adam(list(params), lr=float(hyperparameters.get("sup_lr")), weight_decay=float(hyperparameters.get("sup_wd")))
        else:
            raise ValueError(f"Unsupported supervised optimizer: {SUP_OPT}")

    def make_sup_scheduler(opt: torch.optim.Optimizer):
        if SUP_SCHED == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=META_ITERS, eta_min=float(exp_cfg.get("eta_min")))
        elif SUP_SCHED == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=int(exp_cfg.get("T_0")),
                T_mult=int(exp_cfg.get("T_mult")),
                eta_min=float(exp_cfg.get("eta_min")),
            )
        elif SUP_SCHED in (None, "None"):
            return None
        else:
            raise ValueError(f"Unsupported supervised scheduler: {SUP_SCHED}")

    # -------- episode knobs ---------------
    META_BATCH_SIZE=int(meta_cfg["meta_batch_size"])
    K_SUPPORT = int(meta_cfg["k_support"])
    Q_QUERY = meta_cfg.get("q_query")  # can be None
    Q_EVAL = meta_cfg.get("q_eval")
    INNER_STEPS = int(meta_cfg["inner_steps"])
    INNER_LR = float(meta_cfg["inner_lr"])
    RUN_SIZE = int(meta_cfg["run_size"])
    val = meta_cfg.get("clip_grad_norm")
    CLIP_GRAD = float(val) if val is not None else None
    VAL_EPISODES_PER_SUBJECT = int(meta_cfg["val_episodes_per_subject"])

    # labram-specific shaping knobs used by MetaEngine’s fetch path
    n_patches_labram = int(data_cfg.get("n_patches_labram"))
    
    electrodes = data_cfg.get("electrodes") or get_ku_dataset_channels()

    ### supervised knobs
    supervised_cfg = config.get("supervised")
    n_epochs_supervised = int(supervised_cfg.get("n_epochs_supervised"))
    meta_iters_per_meta_epoch = int(supervised_cfg.get("meta_iters_per_meta_epoch"))
    supervised_train_batch_size = int(supervised_cfg.get("train_batch_size"))
    supervised_eval_batch_size = int(supervised_cfg.get("eval_batch_size"))

    # warm up lazy bits (esp. labram)
    model = model.to(DEVICE)
    n_channels = int(data_cfg.get("n_channels"))
    samples = int(data_cfg.get("samples"))
    with torch.no_grad():
        if model_str == "labram":
            samples = int(data_cfg.get("samples")) // data_cfg.get("n_patches_labram")
            _ = model(
                x=torch.zeros(1, n_channels, n_patches_labram, samples, device=DEVICE),
                electrodes=electrodes,
            )
        else:
            _ = model(x=torch.zeros(1, n_channels, samples, device=DEVICE))

    # -------- engine -----------------------
    engine = MetaEngine(
        # --- BASE ---
        model_str=model_str,
        experiment_name=experiment_name,
        device=DEVICE,
        model=model,
        electrodes=electrodes,
        non_blocking=NON_BLOCKING,
        pin_memory=PIN_MEMORY,
        use_compile=USE_COMPILE,
        use_amp=USE_AMP,
        use_wandb=exp_cfg.get("log_to_wandb"),
        wandb_entity=exp_cfg.get("wandb_entity"),
        wandb_project=exp_cfg.get("wandb_project"),
        config_for_logging=config,
        save_regular_checkpoints=exp_cfg.get("save_regular_checkpoints"),
        save_regular_checkpoints_interval=exp_cfg.get("save_regular_checkpoints_interval"),
        save_best_checkpoints=exp_cfg.get("save_best_checkpoints"),
        save_final_checkpoint=exp_cfg.get("save_final_checkpoint"),
        checkpoint_dir=(Path(__file__).parent / "weights" / "checkpoints_meta" / experiment_name),
        # --- SPECIFIC ---
        meta_iterations=META_ITERS,
        validate_every= int(exp_cfg["validate_every"]),
        validate_meta_every= int(exp_cfg["validate_every"]),
        # datasets / splits 
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        S_train=sm.S_train,
        S_val=sm.S_val,
        S_test=sm.S_test,
        # loss / factories
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_factory=make_optimizer,
        scheduler_factory=make_scheduler,
        # episode design
        meta_batch_size=META_BATCH_SIZE,
        k_support=K_SUPPORT,
        q_query=Q_QUERY,
        q_eval=Q_EVAL,
        val_episodes_per_subject=VAL_EPISODES_PER_SUBJECT,
        inner_steps=INNER_STEPS,
        inner_lr=INNER_LR,
        run_size=RUN_SIZE,
        # RNG 
        seed=SEED,
        clip_grad_norm=CLIP_GRAD,
        # supervised-in-meta knobs
        n_epochs_supervised=n_epochs_supervised,
        meta_iters_per_meta_epoch=meta_iters_per_meta_epoch,
        supervised_train_batch_size=supervised_train_batch_size,
        supervised_eval_batch_size=supervised_eval_batch_size,
        supervised_optimizer_factory=make_sup_optimizer,
        supervised_scheduler_factory=make_sup_scheduler,
    )

    if not with_tester:
        return engine

    tester = TestEngine(
        engine=engine,
        test_ds=test_ds,
        use_wandb=exp_cfg.get("log_to_wandb_test"),
        wandb_prefix="test",
        run_size=100,
        save_dir=Path(config.get("test").get("save_dir_root")) / model_str / experiment_name,
        head_only=hyperparameters.get("head_only_test"),
        test_lr=float(hyperparameters.get("test_lr")),
        test_wd=float(hyperparameters.get("test_wd")),
    )

    return engine, tester
# -------- run --------------------------
if __name__ == "__main__":
    with open("hyperparameters/meta_hyperparameters.yaml", "r") as f:
        config = yaml.safe_load(f)
 
    meta_engine, tester = get_meta_engine(config, with_tester=True)
    meta_engine.train()

    all_results = tester.test_all_subjects(
            shots_list= config.get("test").get("shots"),
            n_epochs= config.get("test").get("n_epochs"),
            n_repeats=config.get("test").get("n_repeats"),
        )