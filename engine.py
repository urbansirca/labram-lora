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
import numpy as np


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
        train_after_stopping_set: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        use_wandb: bool = True,
        electrodes: Optional[List[str]] = None,
        save_checkpoints: bool = True,
        save_checkpoints_interval: int = 10,
        non_blocking: bool = True,
        pin_memory: bool = False,
        use_amp: bool = True,
    ):
        self.hyperparameters = hyperparameters
        self.experiment_name = experiment_name
        self.n_epochs = n_epochs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = torch.compile(model.to(self.device))  # compile the model

        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.train_after_stopping_set = train_after_stopping_set

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = nn.CrossEntropyLoss()

        self.metrics = Metrics()
        self.electrodes = electrodes

        # optimizations
        self.non_blocking = non_blocking
        self.pin_memory = pin_memory

        # wandb
        self.use_wandb = use_wandb and (wandb is not None)
        self.wandb_run = None
        self.config = config
        self.start_time = time.time()
        if self.use_wandb:
            self.wandb_run = self.wandb_setup(config)

        self.save_regular_checkpoints = config["experiment"]["save_regular_checkpoints"]
        self.save_final_checkpoint = config["experiment"]["save_final_checkpoint"]
        self.save_best_checkpoints = config["experiment"]["save_best_checkpoints"]
        if any(
            [
                self.save_regular_checkpoints,
                self.save_final_checkpoint,
                self.save_best_checkpoints,
            ]
        ):
            self.checkpoint_path = self.setup_checkpoint()
            self.save_regular_checkpoints_interval = config["experiment"][
                "save_regular_checkpoints_interval"
            ]

        self.best_val_accuracy = 0.0
        self.best_val_loss = np.inf
        self.best_val_epoch = 0
        self.early_stopping_counter = 0

        self.early_stopping = config["experiment"]["early_stopping"]
        self.early_stopping_patience = config["experiment"]["early_stopping_patience"]
        self.early_stopping_delta = config["experiment"]["early_stopping_delta"]
        self.early_stopping_metric = config["experiment"]["early_stopping_metric"]

        self.train_after_stopping = config["experiment"]["train_after_stopping"]
        self.train_after_stopping_epochs = config["experiment"][
            "train_after_stopping_epochs"
        ]

        self.use_amp = use_amp  # use automatic mixed precision
        self.scaler = torch.amp.GradScaler(device=self.device) if self.use_amp else None

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
            f"{self.checkpoint_path}/{name}",
        )

    def save_regular_checkpoints(self, joined: bool = False):
        if (
            self.save_regular_checkpoints
            and self.metrics.epoch % self.save_regular_checkpoints_interval == 0
        ):
            name = f"checkpoint_{'JOINED' if joined else ''}_e{self.metrics.epoch}_acc{self.metrics.val_accuracy:.3f}.pth"
            self.checkpoint(name=name)

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

    def train_epoch(self, dataloader: DataLoader) -> None:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0, device=self.device)
        total_count = torch.tensor(0, device=self.device)

        for i, (X, Y, sid) in enumerate(dataloader):
            X = X.to(self.device, non_blocking=self.non_blocking)
            Y = Y.long().to(self.device, non_blocking=self.non_blocking)

            self.optimizer.zero_grad(set_to_none=True)

            forward_start_time = time.time()

            if self.use_amp:  # use automatic mixed precision
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    if self.config["experiment"]["model"] == "labram":
                        logits = self.model(x=X, electrodes=self.electrodes)
                    else:
                        logits = self.model(X)

                    backward_start_time = time.time()
                    loss = self.loss_fn(logits, Y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                if self.config["experiment"]["model"] == "labram":
                    logits = self.model(x=X, electrodes=self.electrodes)
                else:
                    logits = self.model(X)

                backward_start_time = time.time()
                loss = self.loss_fn(logits, Y)
                loss.backward()
                self.optimizer.step()

            # Don't call .item() here - just accumulate the loss tensor
            start_time_metrics = time.time()
            bsz = Y.size(0)
            total_loss += loss * bsz  # Keep as tensor
            preds = logits.argmax(dim=1)
            total_correct += (preds == Y).sum()
            total_count += bsz
            logger.debug(
                f"Forward pass time for batch {i}: {backward_start_time - forward_start_time:.2f}s, Backward pass time: {start_time_metrics - backward_start_time:.2f}s, Metrics time: {time.time() - start_time_metrics:.2f}s"
            )

        # Only call .item() once at the end
        self.metrics.train_loss = (total_loss / max(1, total_count)).item()
        self.metrics.train_accuracy = (total_correct / max(1, total_count)).item()
        self.metrics.scheduler_lr = self.optimizer.param_groups[0]["lr"]

        if self.scheduler is not None:
            self.scheduler.step()

    def train(self) -> None:

        for epoch in range(self.n_epochs):
            self.metrics.epoch = epoch + 1
            self.train_epoch(self.training_set)
            self.validate_epoch()
            self.log_metrics()

            if self.early_stopping:
                if self.check_early_stopping():
                    break

            # ---------------- save regular checkpoints ----------------
            self.save_regular_checkpoints(joined=False)

        # ---------------- save the final checkpoint ----------------
        if self.save_final_checkpoint and not self.train_after_stopping:
            name = f"final_checkpoint_e{self.metrics.epoch}_acc{self.metrics.val_accuracy:.3f}.pth"
            self.checkpoint(name=name)

        # ---------------- train after stopping ----------------
        if self.train_after_stopping:
            for epoch in range(self.train_after_stopping_epochs):
                self.metrics.epoch += 1
                self.train_epoch(self.train_after_stopping_set)
                self.log_metrics()

                if self.metrics.val_loss <= self.best_val_loss:
                    logger.info(
                        f"Training after stopping stopped at epoch {epoch} because target loss was reached"
                    )
                    break
                self.save_regular_checkpoints(joined=True)

            self.checkpoint(
                name=f"FINAL_e{self.metrics.epoch}_acc{self.metrics.val_accuracy:.3f}.pth"
            )

    def check_early_stopping(self) -> bool:
        """Check if early stopping criteria is met. Returns True if should stop."""

        if self.early_stopping_metric == "val_loss":
            current_metric = self.metrics.val_loss
            is_better = current_metric < (
                self.best_metric - self.early_stopping_min_delta
            )
        elif self.early_stopping_metric == "val_accuracy":
            current_metric = self.metrics.val_accuracy
            is_better = current_metric > (
                self.best_metric + self.early_stopping_min_delta
            )
        else:
            raise ValueError(
                f"Unknown early stopping monitor: {self.early_stopping_metric}"
            )

        if is_better:
            self.best_metric = current_metric
            self.patience_counter = 0
            logger.info(f"New best {self.early_stopping_metric}: {current_metric:.4f}")

            # Save best model
            if self.save_best_checkpoints:
                self.checkpoint(name="best_val_checkpoint")

            # save these metrics for target loss for training after stopping
            self.best_val_accuracy = self.metrics.val_accuracy
            self.best_val_loss = self.metrics.val_loss
            self.best_val_epoch = self.metrics.epoch
        else:
            self.patience_counter += 1
            logger.info(
                f"Early stopping patience: {self.patience_counter}/{self.early_stopping_patience}"
            )

        if self.patience_counter >= self.early_stopping_patience:
            logger.info(
                f"Early stopping triggered! Best {self.early_stopping_metric}: {self.best_metric:.4f}"
            )
            self.early_stop = True
            return True

        return False

    @torch.no_grad()
    def validate_epoch(self) -> None:
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0, device=self.device)
        total_count = torch.tensor(0, device=self.device)

        for i, (X, Y, sid) in enumerate(self.validation_set):
            X = X.to(self.device, non_blocking=self.non_blocking)
            Y = Y.long().to(self.device, non_blocking=self.non_blocking)

            forward_start_time = time.time()

            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    if self.config["experiment"]["model"] == "labram":
                        logits = self.model(x=X, electrodes=self.electrodes)
                    else:
                        logits = self.model(X)
            else:
                if self.config["experiment"]["model"] == "labram":
                    logits = self.model(x=X, electrodes=self.electrodes)
                else:
                    logits = self.model(X)

            backward_start_time = time.time()
            loss = self.loss_fn(logits, Y)

            start_time_metrics = time.time()
            bsz = Y.size(0)
            total_loss += loss * bsz
            preds = logits.argmax(dim=1)
            total_correct += (preds == Y).sum()
            total_count += bsz

            logger.debug(
                f"Forward pass time for batch {i}: {backward_start_time - forward_start_time:.2f}s, Backward pass time: {start_time_metrics - backward_start_time:.2f}s, Metrics time: {time.time() - start_time_metrics:.2f}s"
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
            X = X.to(self.device, non_blocking=self.non_blocking)
            Y = Y.long().to(self.device, non_blocking=self.non_blocking)

            forward_start_time = time.time()

            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    if self.config["experiment"]["model"] == "labram":
                        logits = self.model(x=X, electrodes=self.electrodes)
                    else:
                        logits = self.model(X)
            else:
                if self.config["experiment"]["model"] == "labram":
                    logits = self.model(x=X, electrodes=self.electrodes)
                else:
                    logits = self.model(X)

            backward_start_time = time.time()
            loss = self.loss_fn(logits, Y)

            start_time_metrics = time.time()
            bsz = Y.size(0)
            total_loss += loss * bsz
            preds = logits.argmax(dim=1)
            total_correct += (preds == Y).sum()
            total_count += bsz

            logger.debug(
                f"Forward pass time for batch {i}: {backward_start_time - forward_start_time:.2f}s, Backward pass time: {start_time_metrics - backward_start_time:.2f}s, Metrics time: {time.time() - start_time_metrics:.2f}s"
            )

        self.metrics.test_loss = (total_loss / max(1, total_count)).item()
        self.metrics.test_accuracy = (total_correct / max(1, total_count)).item()

        # log once for visibility
        self.log_metrics()

    def finish(self):
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
