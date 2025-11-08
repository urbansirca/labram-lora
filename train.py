from collections import defaultdict
import logging
from datetime import datetime
from pathlib import Path
import re
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy

from engines import Engine
from models import load_labram, DeepConvNet, load_labram_with_adapter, freeze_all_but_head_labram, freeze_all_but_head_deepconvnet
from subject_split import KUTrialDataset, SplitConfig, SplitManager
from engines import TestEngine
from preprocessing.preprocess_KU_data import get_ku_dataset_channels, get_dreyer_dataset_channels, get_nikki_dataset_channels


def set_partial_finetune_labram(model, mode="last_k", k=8, verbose=True):
    """
    Freeze everything, then unfreeze:
      - head (always)
      - last k transformer blocks (if mode='last_k')
      - all transformer blocks (if mode='all')
      - head only (if mode='linear_probe')

    Assumes block params are named like 'blocks.<idx>.*' (as your summary shows).
    We intentionally keep embeddings (pos_embed, time_embed, patch_embed, cls_token) FROZEN
    to mirror the paper's 'Transformer blocks' language.
    """
    # 0) freeze all
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 1) find transformer blocks (by name)
    block_map = defaultdict(list)  # idx -> [param names]
    for n, p in model.named_parameters():
        m = re.search(r"(^|\.)(blocks)\.(\d+)\.", n)
        if m:
            idx = int(m.group(3))
            block_map[idx].append(n)

    all_block_ids = sorted(block_map.keys())
    if verbose:
        print(f"[partial-ft] Found {len(all_block_ids)} transformer blocks: {all_block_ids}")
    if not all_block_ids:
        print("[partial-ft][WARN] No blocks.*.* matched; update the regex to your model naming.")
    
    # 2) decide which blocks to unfreeze
    to_unfreeze = set()
    if mode == "all":
        to_unfreeze = set(all_block_ids)
    elif mode == "last_k":
        assert isinstance(k, int) and k > 0, "k must be a positive int"
        to_unfreeze = set(all_block_ids[-k:])
    elif mode == "linear_probe":
        to_unfreeze = set()  # only head below
    else:
        raise ValueError(f"Unknown mode: {mode}")

    name_to_param = dict(model.named_parameters())
    # 3) unfreeze chosen blocks
    for idx in to_unfreeze:
        for pname in block_map.get(idx, []):
            name_to_param[pname].requires_grad = True

    # 4) always unfreeze classification head
    head_hits = 0
    for n, p in model.named_parameters():
        ln = n.lower()
        if ("head." in ln) or ln.endswith("head.weight") or ln.endswith("head.bias") \
           or (".classifier." in ln) or ln.endswith(".classifier.weight") or ln.endswith(".classifier.bias") \
           or ln.endswith(".fc.weight") or ln.endswith(".fc.bias"):
            p.requires_grad = True
            head_hits += 1
    if verbose:
        print(f"[partial-ft] Unfroze head params: {head_hits} tensors")
        print(f"[partial-ft] Unfroze block ids: {sorted(list(to_unfreeze))}")

    # 5) report
    total = sum(p.numel() for _, p in model.named_parameters())
    trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    print("\n========== PARTIAL FINE-TUNING SUMMARY ==========")
    print(f"Mode: {mode}{'' if mode!='last_k' else f' (k={k})'}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,} ({trainable/total:.2%})")
    by_prefix = defaultdict(int)
    for n, p in model.named_parameters():
        if p.requires_grad:
            by_prefix[n.split('.')[0]] += p.numel()
    print("\nBy top-level prefix (trainable):")
    for pref, cnt in sorted(by_prefix.items(), key=lambda x: -x[1]):
        print(f"  {pref:<20} → {cnt:,d} parameters")
    print("-------------------------------------------------\n")
# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def get_engine(config, with_tester = False, experiment_name = None, model = None, model_str = None, model_hyperparameters = None):
    # ---------------- config -----------------


    exp_cfg = config.get("experiment")
    data_cfg = config.get("data")
    samp_cfg = config.get("sampler")
    opt_cfg = config.get("optimizations")

    # core exp settings
    SEED = exp_cfg["seed"]
    N_EPOCHS = exp_cfg["epochs"]
    OPTIMIZER = exp_cfg["optimizer"]
    SCHEDULER = exp_cfg["scheduler"]

    # optimizations
    NUM_WORKERS = opt_cfg.get("num_workers")
    PIN_MEMORY = opt_cfg.get("pin_memory")
    PERSISTENT_WORKERS = opt_cfg.get("persistent_workers")
    NON_BLOCKING = opt_cfg.get("non_blocking")
    USE_AMP = opt_cfg.get("use_amp")

    electrodes = data_cfg.get("electrodes")
    if electrodes is None and "ku" in data_cfg.get("path").lower():
        electrodes = get_ku_dataset_channels()
        logger.info(f"USING KU CHANNELS: {electrodes}")
    else:
        raise ValueError("Please provide electrode list in config for non-KU datasets.")

    # ---------------- model ------------------
    model_name = exp_cfg["model"].lower()
    if model is None:
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
                set_partial_finetune_labram(model, mode="last_k", k=8)
            
            if hyperparameters["head_only_train"]:
                model = freeze_all_but_head_labram(model)
        elif model_name == "deepconvnet":
            hyperparameters = config.get("deepconvnet")
            model_str = "deepconvnet"
            model = DeepConvNet(
                    in_chans=data_cfg.get("input_channels"),
                    n_classes=data_cfg.get("num_classes"),
                    input_time_length=data_cfg.get("samples"),
                    final_conv_length="auto",
                )
            print("MODEL",model)
            if hyperparameters["checkpoint_file"] is not None:
                try:
                    state_dict = torch.load(hyperparameters["checkpoint_file"], map_location="cpu")
                    model.load_state_dict(state_dict)
                    logger.info(f"LOADED DEEPCONVNET FROM {hyperparameters['checkpoint_file']}")
                except Exception as e:
                    # Handle NG models
                    model = DeepConvNet(
                        in_chans=data_cfg.get("input_channels"),
                        n_classes=data_cfg.get("num_classes"),
                        input_time_length=1000,
                        final_conv_length="auto",
                    )
                    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                    torch.serialization.safe_globals([numpy.core.multiarray.scalar])
                    checkpoint = torch.load(hyperparameters["checkpoint_file"], map_location="cpu", weights_only=False)

                    # Extract the checkpoint state dictionary
                    checkpoint_state_dict = checkpoint["model_state_dict"]

                    # Add the "model." prefix to all keys
                    checkpoint_state_dict_renamed = {
                        f"model.{key}" if not key.startswith("model.") else key: value
                        for key, value in checkpoint_state_dict.items()
                    }

                    # Load the renamed state dict into the model
                    model.load_state_dict(checkpoint_state_dict_renamed)
                    logger.info(f"LOADED DEEPCONVNET FROM {hyperparameters['checkpoint_file']} with model_state_dict")


                
            if hyperparameters["head_only_train"]:
                assert hyperparameters["checkpoint_file"] is not None, "When using head_only_train, a checkpoint_file must be specified to load the pretrained weights from."
                model = freeze_all_but_head_deepconvnet(model)
        else:
            raise ValueError("Invalid model")

        logger.info(f"HYPERPARAMETERS for {model_name}: {hyperparameters}")
    
    else:   
        model = model
        model_str = model_str
        hyperparameters = model_hyperparameters
        logger.info(f"USING PROVIDED MODEL: {model_str}")



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
    TRAIN_BS = samp_cfg.get("train_batch_size")
    EVAL_BS = samp_cfg.get("eval_batch_size")
    DROP_LAST = samp_cfg.get("drop_last")
    SHUF_SUBJ = samp_cfg.get("shuffle_subjects")
    SHUF_TRIALS = samp_cfg.get("shuffle_trials")

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


    checkpoint_dir = Path(str(exp_cfg.get("checkpoint_dir") or (Path(__file__).parent / "weights" / "checkpoints" / experiment_name)))

    engine = Engine(
        # --- BASE ---
        model_str=model_str,
        experiment_name=experiment_name,
        device=DEVICE,
        model=model,
        electrodes=electrodes,
        non_blocking=NON_BLOCKING,
        pin_memory=PIN_MEMORY,
        use_amp=USE_AMP,
        use_compile=opt_cfg.get("use_compile"),
        use_wandb=exp_cfg.get("log_to_wandb"),
        wandb_entity="urban-sirca-vrije-universiteit-amsterdam",
        wandb_project=exp_cfg.get("wandb_project"),
        config_for_logging=config,
        save_regular_checkpoints=exp_cfg.get("save_regular_checkpoints"),
        save_regular_checkpoints_interval=exp_cfg.get("save_regular_checkpoints_interval"),
        save_best_checkpoints=exp_cfg.get("save_best_checkpoints"),
        save_final_checkpoint=exp_cfg.get("save_final_checkpoint"),
        checkpoint_dir=checkpoint_dir,
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

    if not with_tester:
        return engine, None
    
    test_cfg = config.get("test")
    save_root = Path(test_cfg.get("save_dir_root", "results/test"))  # ensure Path
    save_dir = save_root / model_str / experiment_name
    tester = TestEngine(
        engine=engine,
        test_ds=test_ds,
        use_wandb=exp_cfg.get("log_to_wandb_test"),
        wandb_prefix="test",
        run_size=100,
        save_dir=save_dir,
        head_only=hyperparameters.get("head_only_test", False),
        test_lr=hyperparameters.get("test_lr"),
        test_wd=hyperparameters.get("test_wd"),
    )
    return engine, tester

    # ---------------- run --------------------
if __name__ == "__main__":
    with open("hyperparameters/hyperparameters.yaml", "r") as f:
        config = yaml.safe_load(f)
 
    engine, tester = get_engine(config, with_tester=True)
    engine.train()

    all_results = tester.test_all_subjects(
            shots_list= config.get("test").get("shots"),
            n_epochs= config.get("test").get("n_epochs"),
            n_repeats=config.get("test").get("n_repeats"),
        )
