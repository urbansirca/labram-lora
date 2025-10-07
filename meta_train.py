import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml
import torch
import torch.nn as nn

from meta_engine import MetaEngine
from models import EEGNet, load_labram, load_labram_with_adapter, DeepConvNet
from subject_split import KUTrialDataset, SplitConfig, SplitManager
from preprocess_KU_data import get_ku_dataset_channels
from test_engine import TestEngine

# -------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



def get_meta_engine(config, with_tester = False, experiment_name = None, model= None, model_str= None, model_hyperparameters = None):
    exp_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    meta_cfg = config.get("meta", {})
    opt_cfg = config.get("optimizations", {})
    peft_cfg = config.get("peft_config", {})
    eegnet_hp = config.get("eegnet", {})
    labram_hp = config.get("labram", {})
    deepconvnet_hp = config.get("deepconvnet", {})
    # core exp
    SEED = int(exp_cfg.get("seed", 111))
    META_ITERS = int(exp_cfg["meta_iterations"])
    VALIDATE_EVERY = int(exp_cfg["validate_every"])
    OPTIMIZER = exp_cfg["optimizer"]
    SCHEDULER = exp_cfg["scheduler"]
    SUP_OPT = OPTIMIZER
    SUP_SCHED = SCHEDULER


    # perf
    USE_AMP = bool(opt_cfg.get("use_amp"))
    USE_COMPILE = bool(opt_cfg.get("use_compile"))
    NON_BLOCKING = bool(opt_cfg.get("non_blocking"))
    PIN_MEMORY = bool(opt_cfg.get("pin_memory"))

    n_channels = int(data_cfg.get("n_channels"))
    samples = int(data_cfg.get("samples"))
    classes = int(data_cfg.get("num_classes"))
    # -------- model ------------
    model_name = exp_cfg["model"].lower()
    if model_name == "labram":
        
        if labram_hp.get("adapter_checkpoint_dir"):
            model = load_labram_with_adapter(
                labram_hp.get("adapter_checkpoint_dir")
            )
        else:
            model = load_labram(
                lora=labram_hp.get("lora"),
                peft_config=peft_cfg,
        )

        model_str = "labram"

        samples = int(data_cfg.get("samples")) // data_cfg.get("n_patches_labram")
    elif model_name == "eegnet":
        model = EEGNet(
            nb_classes=classes,
            Chans=n_channels,
            Samples=samples,
            dropoutRate=eegnet_hp.get("dropoutRate"),
            kernLength=eegnet_hp.get("kernLength"),
            F1=eegnet_hp.get("F1"),
            D=eegnet_hp.get("D"),
            F2=eegnet_hp.get("F2"),
        )
        model_str = "eegnet"

    elif model_name == "deepconvnet":
        model = DeepConvNet(
            in_chans=n_channels,
            n_classes=classes,
            input_time_length=samples,
            final_conv_length="auto",
        )
        model_str = "deepconvnet"
    else:
        raise ValueError("experiment.model must be one of: labram, eegnet")

    # name like train.py
    name_list = [
        f"{exp_cfg['model']}",
        "alternating",
        datetime.now().strftime("%H%M%S"),
    ]
    experiment_name = "_".join(map(str, name_list))
    logger.info(f"Experiment name: {experiment_name}")

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

    # -------- optim/sched factories ----
    if model_str == 'deepconvnet':
        lr = float(deepconvnet_hp.get("lr"))
        wd = float(deepconvnet_hp.get("weight_decay"))
        sup_lr = float(deepconvnet_hp.get("sup_lr"))
        sup_wd = float(deepconvnet_hp.get("sup_wd"))
    elif model_str == 'labram':
        lr = float(labram_hp.get("lr"))
        wd = float(labram_hp.get("weight_decay"))
        sup_lr = float(labram_hp.get("sup_lr"))
        sup_wd = float(labram_hp.get("sup_wd"))
    elif model_str == 'eegnet':
        lr = float(eegnet_hp.get("lr"))
        wd = float(eegnet_hp.get("weight_decay"))
        sup_lr = float(eegnet_hp.get("sup_lr"))
        sup_wd = float(eegnet_hp.get("sup_wd"))
    else:
        raise ValueError(f"Unsupported model: {model_str}")

    if model_str in ['eegnet', 'deepconvnet']:
        def make_optimizer(params: Iterable[torch.nn.Parameter]):
            opt = OPTIMIZER.lower()
            if opt == "adamw":
                return torch.optim.AdamW(list(params), lr=lr, weight_decay=wd)
            elif opt == "adam":
                return torch.optim.Adam(list(params), lr=lr, weight_decay=wd)
            else:
                raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")
    elif model_str == 'labram':
        def make_optimizer(params: Iterable[torch.nn.Parameter]):
            opt = OPTIMIZER.lower()

            allowed_ids = {id(p) for p in params}
            named = [(n, p) for n, p in model.named_parameters() if id(p) in allowed_ids]

            head, lora_or_no_decay = [], []
            for name, p in named:
                if any(k in name.lower() for k in ("head", "classifier", "fc_out", "logits")):
                    head.append(p)
                else:
                    # Any non-head trainable we expect to be LoRA;
                    lora_or_no_decay.append(p)

            base = (labram_hp or eegnet_hp) or {}
            head_lr = float(base.get("head_lr"))
            head_wd = float(base.get("head_weight_decay"))
            lora_lr = float(base.get("lora_lr"))

            groups = []
            if head:
                groups.append({"params": head, "lr": head_lr, "weight_decay": head_wd, "name": "head"})
            if lora_or_no_decay:
                groups.append({"params": lora_or_no_decay, "lr": lora_lr, "weight_decay": 0.0, "name": "lora_no_decay"})
            
            if opt == "adamw":
                # pprint(groups)
                return torch.optim.AdamW(groups)
            elif opt == "adam":
                return torch.optim.Adam(groups)
            else:
                raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")
    else:
        raise ValueError(f"Unsupported model: {model_str}")


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

    def make_optimizer_tester(params: Iterable[torch.nn.Parameter]):
        opt = OPTIMIZER.lower()
        if opt == "adamw":
            return torch.optim.AdamW(list(params), lr=lr, weight_decay=wd)
        elif opt == "adam":
            return torch.optim.Adam(list(params), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

    def make_sup_optimizer(params: Iterable[torch.nn.Parameter]):
        opt = SUP_OPT.lower()
        if opt == "adamw":
            return torch.optim.AdamW(list(params), lr=sup_lr, weight_decay=sup_wd)
        elif opt == "adam":
            return torch.optim.Adam(list(params), lr=sup_lr, weight_decay=sup_wd)
        else:
            raise ValueError(f"Unsupported supervised optimizer: {SUP_OPT}")

    def make_sup_scheduler(opt: torch.optim.Optimizer):
        if SUP_SCHED == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=META_ITERS, eta_min=float(exp_cfg.get("eta_min", 0)))
        elif SUP_SCHED == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=int(exp_cfg.get("T_0", 100)),
                T_mult=int(exp_cfg.get("T_mult", 2)),
                eta_min=float(exp_cfg.get("eta_min", 1e-5)),
            )
        elif SUP_SCHED in (None, "None"):
            return None
        else:
            raise ValueError(f"Unsupported supervised scheduler: {SUP_SCHED}")

    # -------- episode knobs ---------------
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
    patch_len = int(data_cfg.get("patch_length"))
    
    electrodes = data_cfg.get("electrodes") or get_ku_dataset_channels()

    ### supervised knobs
    supervised_cfg = config.get("supervised", {})
    n_epochs_supervised = int(supervised_cfg.get("n_epochs_supervised"))
    meta_iters_per_meta_epoch = int(supervised_cfg.get("meta_iters_per_meta_epoch"))
    supervised_train_batch_size = int(supervised_cfg.get("train_batch_size"))
    supervised_eval_batch_size = int(supervised_cfg.get("eval_batch_size"))

    # warm up lazy bits (esp. labram)
    model = model.to(DEVICE)
    with torch.no_grad():
        if model_str == "labram":
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
        save_regular_checkpoints_interval=exp_cfg.get("save_regular_checkpoints_interval", 10),
        save_best_checkpoints=exp_cfg.get("save_best_checkpoints"),
        save_final_checkpoint=exp_cfg.get("save_final_checkpoint"),
        checkpoint_dir=(Path(__file__).parent / "weights" / "checkpoints_meta" / experiment_name),
        # --- SPECIFIC ---
        meta_iterations=META_ITERS,
        validate_every= VALIDATE_EVERY,
        validate_meta_every= VALIDATE_EVERY,
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
        use_wandb=exp_cfg.get("log_to_wandb"),
        wandb_prefix="test",
        run_size = 100,
    )

    return engine, tester
# -------- run --------------------------
if __name__ == "__main__":
    # -------- config -----------
    with open("hyperparameters/meta_hyperparameters.yaml", "r") as f:
        config = yaml.safe_load(f)

    engine, tester = get_meta_engine(config, with_tester=True)

    test_cfg = config.get("test", {})
    try:
        engine.train_alternating()
        all_results = tester.test_all_subjects(
            shots_list= [0, 1, 2],
            n_epochs= 2,
        )
    finally:
        engine.train_ds.close()
        engine.val_ds.close()
        engine.test_ds.close()