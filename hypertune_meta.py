import optuna
import optuna.pruners
from optuna.integration.wandb import WeightsAndBiasesCallback
import logging
import yaml
from datetime import datetime
from pathlib import Path
from meta_train import get_meta_engine
import copy

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


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def objective(trial):
    """Optuna objective function."""

    logger.info(f"=" * 60)
    logger.info(f"STARTING TRIAL {trial.number}")
    logger.info(f"=" * 60)

    # Load configs
    with open("hyperparameters/meta_hyperparameters.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    cfg = copy.deepcopy(base_config)

    trial_overrides = {
        "meta": {
            "meta_batch_size": trial.suggest_int("meta_batch_size", 8, 20),
            "k_support": trial.suggest_int("k_support", 1, 5),
            "q_query": trial.suggest_int("q_query", 20, 40),
            "inner_steps": trial.suggest_int("inner_steps", 1, 3),
            "inner_lr": trial.suggest_float("inner_lr", 1e-5, 1e-2, log=True),
        },
        "labram": {
            "lora_lr": trial.suggest_float("lora_lr", 1e-5, 5e-4, log=True),
            "head_lr": trial.suggest_float("head_lr", 1e-5, 1e-3, log=True),
            # "head_weight_decay": trial.suggest_float("head_weight_decay", 0.0, 1e-3, log=True),
        },
        "experiment": {
            "meta_iterations": cfg["experiment"].get("meta_iterations", 200),
            "validate_every": max(1, cfg["experiment"].get("validate_every", 20)),
            "save_final_checkpoint": False,
            "save_regular_checkpoints": False,
            "log_to_wandb": False,
        },
    }

    deep_update(cfg, trial_overrides)
    engine = get_meta_engine(cfg, with_tester=False)


    # Log key hyperparameters
    logger.info(f"Key hyperparameters:")
    logger.info(trial_overrides)

    try:
        logger.info("Starting training...")
        engine.train()

        if engine.metrics.val_accuracy is None:
            engine.meta_validate_epoch()

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
        engine.train_ds.close()
        engine.val_ds.close()
        engine.test_ds.close()


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
