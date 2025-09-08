import logging
from typing import Any, Dict, Optional, Callable, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
import pathlib
import yaml
import wandb


logger = logging.getLogger(__name__)


class Metrics:
    epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    support_loss: Optional[float] = None
    query_loss: Optional[float] = None
    scheduler_lr: Optional[float] = None


class Engine:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        experiment_name: str,
        n_epochs: int,
        device: str,
        training_set: DataLoader,
        validation_set: DataLoader,
        test_set: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        use_wandb: bool = True,
        electrodes: Optional[List[str]] = None,
        save_checkpoints: bool = True,
        save_checkpoints_interval: int = 10,
        non_blocking: bool = True,
        pin_memory: bool = False,
    ):
        self.hyperparameters = hyperparameters
        self.experiment_name = experiment_name
        self.n_epochs = n_epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = Metrics()
        self.electrodes = electrodes
        self.non_blocking = non_blocking
        self.pin_memory = pin_memory
        self.use_wandb = use_wandb and (wandb is not None)
        self.wandb_run = None
        self.config = config
        self.start_time = time.time()
        if self.use_wandb:
            self.wandb_run = self.wandb_setup(config)

        self.save_checkpoints = save_checkpoints
        if self.save_checkpoints:
            self.checkpoint_path = self.setup_checkpoint()
            self.save_checkpoints_interval = save_checkpoints_interval
        self.best_val_accuracy = (
            0.0  # used for checkpointing, TODO: implement early stopping
        )

    def setup_optimizations(self):
        torch.backends.cudnn.benchmark = True

        # Optional: Set memory format for better performance on modern GPUs
        # (Only if your model supports channels-last)
        # torch.backends.cudnn.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_tf32 = True

    def setup_checkpoint(self):
        """Setup checkpoint of the model."""
        path = (
            pathlib.Path(__file__).parent
            / "weights"
            / "checkpoints"
            / self.experiment_name
        )
        path.mkdir(parents=True, exist_ok=True)
        # save the whle config
        with open(path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

        return path

    def checkpoint(self, name: str = "checkpoint"):
        """Save checkpoint of the model."""
        torch.save(
            self.model.state_dict(),
            f"{self.checkpoint_path}/model_{name}_{self.metrics.epoch}.pth",
        )

    def wandb_setup(self, config: Dict[str, Any]):
        logger.info(f"Setting up wandb with experiment name: {self.experiment_name}")
        return wandb.init(
            entity="urban-sirca-vrije-universiteit-amsterdam",
            project="EEG-FM",
            name=self.experiment_name,
            config=config,
        )

    def log_metrics(self):
        """Send metrics to console (always) and wandb (if enabled)."""
        present = {k: v for k, v in self.metrics.__dict__.items() if v is not None}

        # console
        line = [
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in present.items()
        ]

        line.append(f"Runtime: {time.time() - self.start_time:.2f}s")
        self.start_time = time.time()

        logger.info(" | ".join(line))

        # wandb
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log(present)

    def train_epoch(self) -> None:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0, device=self.device)
        total_count = torch.tensor(0, device=self.device)

        for i, (X, Y, sid) in enumerate(self.training_set):
            X = X.to(self.device, non_blocking=self.non_blocking)
            Y = Y.long().to(self.device, non_blocking=self.non_blocking)

            self.optimizer.zero_grad(set_to_none=True)

            forward_start_time = time.time()
            if self.config["experiment"]["model"] == "labram":
                logits = self.model(x=X, electrodes=self.electrodes)
            else:
                logits = self.model(X)
            logger.debug(
                f"Forward pass time for batch {i}: {time.time() - forward_start_time:.2f}s"
            )

            backward_start_time = time.time()
            loss = self.loss_fn(logits, Y)
            loss.backward()
            self.optimizer.step()
            logger.debug(
                f"Backward pass time for batch {i}: {time.time() - backward_start_time:.2f}s"
            )

            # Don't call .item() here - just accumulate the loss tensor
            start_time_metrics = time.time()
            bsz = Y.size(0)
            total_loss += loss * bsz  # Keep as tensor
            preds = logits.argmax(dim=1)
            total_correct += (preds == Y).sum()
            total_count += bsz
            logger.debug(
                f"Metrics time for batch {i}: {time.time() - start_time_metrics:.2f}s"
            )

        # Only call .item() once at the end
        self.metrics.train_loss = (total_loss / max(1, total_count)).item()
        self.metrics.train_accuracy = (total_correct / max(1, total_count)).item()
        self.metrics.scheduler_lr = self.optimizer.param_groups[0]["lr"]

        if self.scheduler is not None:
            self.scheduler.step()

    @torch.no_grad()
    def validate_epoch(self) -> None:
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0, device=self.device)
        total_count = torch.tensor(0, device=self.device)

        for i, (X, Y, sid) in enumerate(self.validation_set):
            start_time_validate = time.time()
            X = X.to(self.device, non_blocking=self.non_blocking)
            Y = Y.long().to(self.device, non_blocking=self.non_blocking)

            logits = self.model(x=X, electrodes=self.electrodes)
            loss = self.loss_fn(logits, Y)
            logger.debug(
                f"Validation loss time for batch {i}: {time.time() - start_time_validate:.2f}s"
            )
            start_time_metrics = time.time()
            bsz = Y.size(0)
            total_loss += loss * bsz
            preds = logits.argmax(dim=1)
            total_correct += (preds == Y).sum()
            total_count += bsz

            logger.debug(
                f"Metrics time for batch {i}: {time.time() - start_time_metrics:.2f}s"
            )

        self.metrics.val_loss = (total_loss / max(1, total_count)).item()
        self.metrics.val_accuracy = (total_correct / max(1, total_count)).item()

    @torch.no_grad()
    def test(self) -> None:
        if self.test_set is None:
            logger.info("No test_set provided; skipping test().")
            return
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0, device=self.device)
        total_count = torch.tensor(0, device=self.device)

        for i, (X, Y, sid) in enumerate(self.test_set):
            start_time_test = time.time()
            X = X.to(self.device, non_blocking=self.non_blocking)
            Y = Y.long().to(self.device, non_blocking=self.non_blocking)

            logits = self.model(x=X, electrodes=self.electrodes)
            loss = self.loss_fn(logits, Y)
            logger.debug(
                f"Test loss time for batch {i}: {time.time() - start_time_test:.2f}s"
            )
            start_time_metrics = time.time()

            bsz = Y.size(0)
            total_loss += loss * bsz
            preds = logits.argmax(dim=1)
            total_correct += (preds == Y).sum()
            total_count += bsz

            logger.debug(
                f"Metrics time for batch {i}: {time.time() - start_time_metrics:.2f}s"
            )

        self.metrics.test_loss = (total_loss / max(1, total_count)).item()
        self.metrics.test_accuracy = (total_correct / max(1, total_count)).item()

        # log once for visibility
        self.log_metrics()

    def train(self) -> None:
        for epoch in range(self.n_epochs):
            self.metrics.epoch = epoch + 1
            self.train_epoch()
            self.validate_epoch()
            self.log_metrics()

            if (
                self.metrics.val_accuracy > self.best_val_accuracy
                and self.metrics.epoch > self.save_checkpoints_interval
            ):
                self.best_val_accuracy = self.metrics.val_accuracy
                self.checkpoint(name="best_val_checkpoint")
            if (
                self.save_checkpoints
                and self.metrics.epoch % self.save_checkpoints_interval == 0
            ):
                self.checkpoint(name="checkpoint")

        # save the final checkpoint
        self.checkpoint(name="final_checkpoint")

    def finish(self):
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
