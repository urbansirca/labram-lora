import optuna
import optuna.pruners
from optuna.integration.wandb import WeightsAndBiasesCallback
import logging
import yaml
import torch
from datetime import datetime
from pathlib import Path

from meta_engine import MetaEngine
from models import load_labram, load_labram_with_adapter
from subject_split import KUTrialDataset, SplitConfig, SplitManager
from preprocess_KU_data import get_ku_dataset_channels

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SaveBestParamsCallback:
    def __call__(self, study, trial):
        # Save best params after each trial
        if study.best_trial.number == trial.number:  # Only save if this trial was the best
            Path("optuna").mkdir(parents=True, exist_ok=True)
            with open("optuna/best_hyperparameters.yaml", "w") as f:
                yaml.dump(study.best_params, f)
            logger.info(f"Updated best hyperparameters after trial {trial.number}: {study.best_value:.4f}")

def load_model_from_checkpoint(checkpoint_path, lora_dropout=0.5, device="cpu"):
    """Load model from checkpoint or create fresh model if None."""
    if checkpoint_path is None:
        # Fresh model with LoRA
        logger.info(f"Loading fresh model with LoRA dropout={lora_dropout}")
        return load_labram(
            lora=True,
            peft_config={
                "r": 2,
                "lora_alpha": 32,
                "lora_dropout": lora_dropout,
                "target_modules": ["qkv", "fc1", "proj"],
            },
        )

    # Load from checkpoint
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    if (Path(checkpoint_path) / "adapter_config.json").exists():
        return load_labram_with_adapter(checkpoint_path, device)
    else:
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")

def suggest_param(trial, name, config_value):
    """Suggest parameter based on config value."""

    # Special cases: these parameters use exact values only, never ranges
    categorical_params = {"lora_dropout", "optimizer", "scheduler", "q_query"}

    if isinstance(config_value, list):
        if len(config_value) == 1:
            # Single value - don't tune, just return it
            value = config_value[0]
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
            return value
        elif len(config_value) == 2:
            # Two values - check if special case or range
            if name in categorical_params:
                # Force categorical for special parameters
                return trial.suggest_categorical(name, config_value)

            val1, val2 = config_value[0], config_value[1]

            # Try to convert strings to numbers first
            try:
                if isinstance(val1, str):
                    val1_num = float(val1)
                else:
                    val1_num = val1
                if isinstance(val2, str):
                    val2_num = float(val2)
                else:
                    val2_num = val2

                # If both are integers, use integer range
                if isinstance(val1, int) and isinstance(val2, int):
                    return trial.suggest_int(name, min(val1, val2), max(val1, val2))
                else:
                    # Use float range for scientific notation and floats
                    low, high = min(val1_num, val2_num), max(val1_num, val2_num)
                    if low > 0 and high / low > 10:
                        return trial.suggest_float(name, low, high, log=True)
                    else:
                        return trial.suggest_float(name, low, high)

            except (ValueError, TypeError):
                # If conversion fails, treat as categorical
                return trial.suggest_categorical(name, config_value)

        elif len(config_value) == 3:
            # Three values - check if it's [low, high, step] for discrete uniform
            try:
                low, high, step = (
                    float(config_value[0]),
                    float(config_value[1]),
                    float(config_value[2]),
                )
                return trial.suggest_discrete_uniform(name, low, high, step)
            except (ValueError, TypeError):
                # If not numeric, treat as categorical
                return trial.suggest_categorical(name, config_value)
        else:
            # Multiple values - categorical
            return trial.suggest_categorical(name, config_value)
    else:
        # Single value - don't tune
        value = config_value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return value


def objective(trial):
    """Optuna objective function."""

    logger.info(f"=" * 60)
    logger.info(f"STARTING TRIAL {trial.number}")
    logger.info(f"=" * 60)

    # Load configs
    with open("meta_hyperparameters.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    with open("hyperparam_meta_tuning.yaml", "r") as f:
        tune_config = yaml.safe_load(f)

    # Suggest hyperparameters
    params = {}

    # Labram hyperparameters
    labram_config = tune_config.get("labram", {})
    params["lr"] = suggest_param(trial, "lr", labram_config.get("lr", 1e-4))
    params["weight_decay"] = suggest_param(
        trial, "weight_decay", labram_config.get("weight_decay", 0.01)
    )

    # Meta hyperparameters
    meta_config = tune_config.get("meta", {})
    params["meta_batch_size"] = suggest_param(
        trial, "meta_batch_size", meta_config.get("meta_batch_size", 4)
    )
    params["k_support"] = suggest_param(
        trial, "k_support", meta_config.get("k_support", 5)
    )
    params["inner_steps"] = suggest_param(
        trial, "inner_steps", meta_config.get("inner_steps", 3)
    )
    params["inner_lr"] = suggest_param(
        trial, "inner_lr", meta_config.get("inner_lr", 1e-3)
    )
    params["steps_per_epoch"] = suggest_param(
        trial, "steps_per_epoch", meta_config.get("steps_per_epoch", 5)
    )
    params["q_query"] = suggest_param(trial, "q_query", meta_config.get("q_query", 100))
    params["run_size"] = suggest_param(
        trial, "run_size", meta_config.get("run_size", 100)
    )

    # Experiment hyperparameters
    exp_config = tune_config.get("experiment", {})
    params["epochs"] = suggest_param(trial, "epochs", exp_config.get("epochs", 5))
    params["optimizer"] = suggest_param(
        trial, "optimizer", exp_config.get("optimizer", "AdamW")
    )
    params["scheduler"] = suggest_param(
        trial, "scheduler", exp_config.get("scheduler", "CosineAnnealingLR")
    )

    # PEFT config
    peft_config = tune_config.get("peft_config", {})
    params["lora_dropout"] = suggest_param(
        trial, "lora_dropout", peft_config.get("lora_dropout", 0.3)
    )

    # Data config
    data_config = tune_config.get("data", {})
    params["n_patches_labram"] = suggest_param(
        trial, "n_patches_labram", data_config.get("n_patches_labram", 4)
    )
    params["patch_length"] = suggest_param(
        trial, "patch_length", data_config.get("patch_length", 200)
    )
    params["input_channels"] = suggest_param(
        trial, "input_channels", data_config.get("input_channels", 62)
    )
    params["num_classes"] = suggest_param(
        trial, "num_classes", data_config.get("num_classes", 2)
    )

    # Checkpoint selection
    checkpoints_config = tune_config.get("checkpoints", {})
    available_paths = checkpoints_config.get("available_paths")
    if available_paths is None:
        available_paths = []
    checkpoint_choices = available_paths # + [None]
    params["checkpoint_path"] = trial.suggest_categorical(
        "checkpoint_path", checkpoint_choices
    )

    # Log key hyperparameters
    logger.info(f"Key hyperparameters:")
    logger.info(f"  lr: {params['lr']}")
    logger.info(f"  meta_batch_size: {params['meta_batch_size']}")
    logger.info(f"  k_support: {params['k_support']}")
    logger.info(f"  inner_steps: {params['inner_steps']}")
    logger.info(f"  inner_lr: {params['inner_lr']}")
    logger.info(f"  epochs: {params['epochs']}")
    logger.info(f"  lora_dropout: {params['lora_dropout']}")
    logger.info(f"  checkpoint_path: {params['checkpoint_path']}")

    # Fixed settings from base config
    data_cfg = base_config.get("data", {})
    exp_cfg = base_config.get("experiment", {})
    opt_cfg = base_config.get("optimizations", {})

    SEED = exp_cfg.get("seed", 111)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # Set seeds
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Data setup
    DATASET_PATH = data_config.get("path", data_cfg["path"])
    SUBJECT_IDS = data_config.get("subjects") or range(
        1, exp_cfg.get("n_subjects", 54) + 1
    )
    TRAIN_PROP = data_config.get("train_proportion", 0.9)
    LEAVE_OUT = data_config.get("leave_out")
    M_LEAVE_OUT = data_config.get("m_leave_out")

    logger.info(f"Dataset setup:")
    logger.info(f"  Path: {DATASET_PATH}")
    logger.info(
        f"  Train subjects: {len([s for s in SUBJECT_IDS if s not in (LEAVE_OUT or [])])}"
    )
    logger.info(f"  Test subjects: {LEAVE_OUT}")

    split_cfg = SplitConfig(
        subject_ids=SUBJECT_IDS,
        m_leave_out=M_LEAVE_OUT,
        subject_ids_leave_out=LEAVE_OUT,
        train_proportion=TRAIN_PROP,
        seed=SEED,
    )
    sm = SplitManager(split_cfg)

    # Create datasets
    train_ds = KUTrialDataset(DATASET_PATH, sm.S_train)
    val_ds = KUTrialDataset(DATASET_PATH, sm.S_val)
    test_ds = KUTrialDataset(DATASET_PATH, sm.S_test)

    try:
        # Load model
        logger.info("Loading model...")
        model = load_model_from_checkpoint(
            params["checkpoint_path"],
            lora_dropout=params["lora_dropout"],
            device=str(DEVICE),
        )
        model = model.to(DEVICE)

        # Warm up model
        electrodes = get_ku_dataset_channels()
        logger.info("Warming up model...")
        with torch.no_grad():
            _ = model(
                x=torch.zeros(
                    1,
                    int(params["input_channels"]),
                    int(params["n_patches_labram"]),
                    int(params["patch_length"]),
                    device=DEVICE,
                ),
                electrodes=electrodes,
            )

        # Optimizer factory
        def make_optimizer(model_params):
            if params["optimizer"] == "AdamW":
                return torch.optim.AdamW(
                    list(model_params),
                    lr=params["lr"],
                    weight_decay=params["weight_decay"],
                )
            else:
                return torch.optim.Adam(
                    list(model_params),
                    lr=params["lr"],
                    weight_decay=params["weight_decay"],
                )

        # Scheduler factory
        def make_scheduler(opt):
            if params["scheduler"] == "CosineAnnealingLR":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=int(params["epochs"])
                )
            else:
                return None

        # Create experiment name
        experiment_name = (
            f"optuna_trial_{trial.number}_{datetime.now().strftime('%H%M%S')}"
        )
        logger.info(f"Experiment name: {experiment_name}")

        # Create MetaEngine
        logger.info("Creating MetaEngine...")
        engine = MetaEngine(
            model=model,
            model_str="labram",
            experiment_name=experiment_name,
            device=DEVICE,
            n_epochs=int(params["epochs"]),
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            S_train=sm.S_train,
            S_val=sm.S_val,
            S_test=sm.S_test,
            loss_fn=None,
            optimizer_factory=make_optimizer,
            scheduler_factory=make_scheduler,
            meta_batch_size=int(params["meta_batch_size"]),
            k_support=int(params["k_support"]),
            q_query=int(params["q_query"]),
            inner_steps=int(params["inner_steps"]),
            inner_lr=params["inner_lr"],
            steps_per_epoch=int(params["steps_per_epoch"]),
            run_size=int(params["run_size"]),
            use_amp=opt_cfg.get("use_amp", False),
            non_blocking=opt_cfg.get("non_blocking", False),
            pin_memory=opt_cfg.get("pin_memory", False),
            use_compile=opt_cfg.get("use_compile", False),
            use_wandb=tune_config.get("optimization", {}).get(
                "use_wandb_trials", False
            ),  # Individual trial logging
            early_stopping=exp_cfg.get("early_stopping", True),
            early_stopping_patience=exp_cfg.get("early_stopping_patience", 3),
            early_stopping_delta=exp_cfg.get("early_stopping_delta", 0.0),
            seed=SEED,
            n_patches_labram=int(params["n_patches_labram"]),
            samples=int(params["patch_length"]),
            channels=int(params["input_channels"]),
            electrodes=electrodes,
        )

        # Train
        logger.info("Starting training...")
        engine.train()

        # Return validation accuracy
        val_accuracy = engine.metrics.val_accuracy
        train_accuracy = engine.metrics.train_accuracy
        val_loss = engine.metrics.val_loss

        logger.info(f"Training completed!")
        logger.info(f"  Final train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Final val accuracy: {val_accuracy:.4f}")
        logger.info(f"  Final val loss: {val_loss:.4f}")

        if val_accuracy is None:
            logger.warning("No validation accuracy available, returning 0.0")
            return 0.0

        # Store additional metrics
        trial.set_user_attr("train_accuracy", train_accuracy)
        trial.set_user_attr("val_loss", val_loss)

        logger.info(
            f"TRIAL {trial.number} COMPLETED - Val Accuracy: {val_accuracy:.4f}"
        )
        return val_accuracy

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 0.0

    finally:
        train_ds.close()
        val_ds.close()
        test_ds.close()


def main():
    """Run optuna optimization."""

    logger.info("=" * 80)
    logger.info("STARTING OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)

    # Load optimization config
    with open("hyperparam_meta_tuning.yaml", "r") as f:
        config = yaml.safe_load(f)

    opt_config = config.get("optimization", {})

    # Create unique study name to avoid database conflicts
    study_name = f"labram_meta_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Study name: {study_name}")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage="sqlite:///optuna_labram_study.db",
        load_if_exists=True,
        pruner=(
            optuna.pruners.MedianPruner()
            if opt_config.get("use_pruning", True)
            else optuna.pruners.NopPruner()
        ),
    )

    # Setup WandB callback if enabled
    callbacks = []
    use_wandb = opt_config.get("use_wandb", False)
    if use_wandb:
        logger.info("WandB integration enabled")
        wandb_project = opt_config.get("wandb_project", "labram-optuna-optimization")
        wandb_entity = opt_config.get("wandb_entity")  # Optional

        wandb_kwargs = {"project": wandb_project}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity

        wandb_callback = WeightsAndBiasesCallback(
            metric_name="val_accuracy", wandb_kwargs=wandb_kwargs
        )
        callbacks.append(wandb_callback)
        logger.info(f"WandB project: {wandb_project}")
        if wandb_entity:
            logger.info(f"WandB entity: {wandb_entity}")
    else:
        logger.info("WandB integration disabled")
    
    callbacks.append(SaveBestParamsCallback())

    logger.info("Configuration:")
    logger.info(f"  Number of trials: 50")
    logger.info(f"  Timeout: 6 hours")
    logger.info(f"  Pruning enabled: {opt_config.get('use_pruning', True)}")
    logger.info(f"  Direction: maximize validation accuracy")

    # Run optimization
    logger.info("Starting optimization...")
    study.optimize(objective, n_trials=50, timeout=3600 * 6, callbacks=callbacks)

    # Print results
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("=" * 80)
    print("=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best parameters
    Path("optuna").mkdir(parents=True, exist_ok=True)
    with open("optuna/best_hyperparameters.yaml", "w") as f:
        yaml.dump(study.best_params, f)

    logger.info(f"Best hyperparameters saved to optuna/best_hyperparameters.yaml")

    return study


if __name__ == "__main__":
    main()
