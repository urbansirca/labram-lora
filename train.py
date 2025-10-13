import logging
from datetime import datetime
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy

from engine import Engine
from models import EEGNet, load_labram, DeepConvNet, load_labram_with_adapter, freeze_all_but_head_labram, freeze_all_but_head_deepconvnet
from subject_split import KUTrialDataset, SplitConfig, SplitManager
from test_engine import TestEngine
from preprocessing.preprocess_KU_data import get_ku_dataset_channels, get_dreyer_dataset_channels, get_nikki_dataset_channels


# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def get_engine(config, with_tester = False, experiment_name = None, model = None, model_str = None, model_hyperparameters = None):
    # ---------------- config -----------------


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

    electrodes = data_cfg.get("electrodes", None)
    if electrodes is None and "dreyer" in data_cfg.get("path", "").lower():
        electrodes = get_dreyer_dataset_channels()  # Dreyer uses the same 32 channels as KU
        logger.info(f"USING DREYER 27 CHANNELS: {electrodes}")
        
    elif electrodes is None and "ku" in data_cfg.get("path", "").lower():
        electrodes = get_ku_dataset_channels()
        logger.info(f"USING KU CHANNELS: {electrodes}")
    
    elif electrodes is None and "nikki" in data_cfg.get("path", "").lower():
        electrodes = get_nikki_dataset_channels()
        logger.info(f"USING NIKKI CHANNELS: {electrodes}")
    else:
        raise ValueError("Please provide electrode list in config for non-KU datasets.")
    # logger.info(f"USING ELECTRODES: {electrodes}")

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
            
            if hyperparameters["head_only_train"]:
                model = freeze_all_but_head_labram(model)
                

        elif model_name == "deepconvnet":
            hyperparameters = config.get("deepconvnet", {})
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
                        in_chans=data_cfg.get("input_channels", 62),
                        n_classes=data_cfg.get("num_classes", 2),
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

        elif model_name == "eegnet":
            hyperparameters = config.get("eegnet", {})
            chans = data_cfg.get("input_channels", 62)
            samples = data_cfg.get("samples", 800)
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
            print("MODEL",model)
            model_str = "eegnet"
            
        elif model_name == "mirepnet":
            hyperparameters = config.get("mirepnet", {})
            model_str = "mirepnet"
            from models.mirepnet import make_mirepnet
            model = make_mirepnet(
                use_lora=hyperparameters.get("use_lora"),
                lora_config=config.get("peft_config"),
            )
            if hyperparameters["head_only_train"]:
                logger.warning("head_only_train is not implemented for MIREPNet yet.")
                raise NotImplementedError
            
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
    train_ds = KUTrialDataset(DATASET_PATH, sm.S_train, select_mirepnet_45=(model_str=="mirepnet"))
    val_ds = KUTrialDataset(DATASET_PATH, sm.S_val, select_mirepnet_45=(model_str=="mirepnet"))
    test_ds = KUTrialDataset(DATASET_PATH, sm.S_test, select_mirepnet_45=(model_str=="mirepnet"))
    train_after_stopping_ds = KUTrialDataset(DATASET_PATH, sm.S_train + sm.S_val, select_mirepnet_45=(model_str=="mirepnet"))


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
                T_0=N_EPOCHS // exp_cfg.get("T_0", 2),
                T_mult=exp_cfg.get("T_mult", 2),
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
        use_compile=opt_cfg.get("use_compile", False),
        use_wandb=exp_cfg.get("log_to_wandb"),
        wandb_entity="urban-sirca-vrije-universiteit-amsterdam",
        wandb_project=exp_cfg.get("wandb_project", "EEG-FM"),
        config_for_logging=config,
        save_regular_checkpoints=exp_cfg.get("save_regular_checkpoints", False),
        save_regular_checkpoints_interval=exp_cfg.get("save_regular_checkpoints_interval", 10),
        save_best_checkpoints=exp_cfg.get("save_best_checkpoints", True),
        save_final_checkpoint=exp_cfg.get("save_final_checkpoint", True),
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
        early_stopping=exp_cfg.get("early_stopping", True),
        early_stopping_patience=exp_cfg.get("early_stopping_patience", 10),
        early_stopping_delta=exp_cfg.get("early_stopping_delta", 0.0),
        # train-after-stopping
        train_after_stopping=exp_cfg.get("train_after_stopping", False),
        train_after_stopping_epochs=exp_cfg.get("train_after_stopping_epochs", 0),
    )

    if not with_tester:
        return engine, None
    
    
    tester = TestEngine(
        engine=engine,
        test_ds=test_ds,
        use_wandb=exp_cfg.get("log_to_wandb_test", False),
        wandb_prefix="test",
        run_size=100,
        save_dir=exp_cfg.get("test", {}).get("save_dir_root", Path("results/test")) / model_str / experiment_name,
        head_only=hyperparameters.get("head_only_test"),
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
