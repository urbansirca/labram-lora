import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import wandb

from base_engine import BaseEngine

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


class Engine(BaseEngine):
    def __init__(
        self,
        # --- BASE ---
        model_str: str,
        experiment_name: str,
        device: Union[str, torch.device],
        model: nn.Module,
        electrodes: Optional[List[str]] = None,
        *,
        non_blocking: bool,
        pin_memory: bool,
        use_amp: bool,
        use_compile: bool,
        use_wandb: bool = False,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        config_for_logging: Optional[Dict[str, Any]] = None,
        save_regular_checkpoints: bool = False,
        save_regular_checkpoints_interval: int = 10,
        save_best_checkpoints: bool = True,
        save_final_checkpoint: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        # --- SPECIFIC ---
        n_epochs: int,
        # data loaders
        training_set: DataLoader,
        validation_set: DataLoader,
        test_set: Optional[DataLoader],
        train_after_stopping_set: Optional[DataLoader],
        # loss / factories
        loss_fn: Optional[nn.Module],
        optimizer_factory: Callable[[nn.Module], Optimizer],
        scheduler_factory: Callable[[Optimizer], Optional[_LRScheduler]],
        # explicit shape checks
        input_channels: int,
        trial_length: int,
        n_patches_labram: Optional[int] = None,
        patch_length: Optional[int] = None,
        # early stopping
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.0,
        # train-after-stopping
        train_after_stopping: bool = False,
        train_after_stopping_epochs: int = 0,
    ):
        super().__init__(
            model_str=model_str,
            experiment_name=experiment_name,
            device=device,
            model=model,
            electrodes=electrodes,
            non_blocking=non_blocking,
            pin_memory=pin_memory,
            use_amp=use_amp,
            use_compile=use_compile,
            use_wandb=use_wandb,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            config_for_logging=config_for_logging,
            save_regular_checkpoints=save_regular_checkpoints,
            save_regular_checkpoints_interval=save_regular_checkpoints_interval,
            save_best_checkpoints=save_best_checkpoints,
            save_final_checkpoint=save_final_checkpoint,
            checkpoint_dir=checkpoint_dir,
        )

        self.n_epochs = int(n_epochs)

        # data loaders
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.train_after_stopping_set = train_after_stopping_set

        # loss / factories
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.optimizer = self.optimizer_factory(self.model)
        self.scheduler = self.scheduler_factory(self.optimizer)

        # explicit shape checks
        self.input_channels = input_channels
        self.trial_length = trial_length
        self.n_patches_labram = n_patches_labram
        self.patch_length = patch_length

        # early stopping
        self.early_stopping = early_stopping
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_delta = float(early_stopping_delta)

        # train-after-stopping
        self.train_after_stopping = train_after_stopping
        self.train_after_stopping_epochs = int(train_after_stopping_epochs)

        # metrics
        self.metrics = Metrics()
        self.best_val_accuracy = 0.0
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0
        self.patience_counter = 0

        self.assert_dimensions()

    # assert dimensions of the dataset are correct given model
    def assert_dimensions(self):
        """Asserts the dimensions of the dataloader"""

        Xshape = next(iter(self.training_set))[0].shape

        _, C, *rest = Xshape
        if self.model_str == "eegnet":
            return
        elif self.model_str == "labram":
            assert self.n_patches_labram is not None and self.patch_length is not None, \
                "labram requires n_patches_labram and patch_length"
            assert (C, *rest) == (self.input_channels, self.n_patches_labram, self.patch_length), \
                f"Per-sample shape should be {(self.input_channels, self.n_patches_labram, self.patch_length)} but is {(C, *rest)}"


    def save_regular_checkpoint(self, joined: bool = False):
        if self.save_regular_checkpoints and self.metrics.epoch % self.save_regular_checkpoints_interval == 0:
            joined_txt = "JOINED" if joined else ""
            metric_val = self.metrics.val_accuracy if not joined else self.metrics.train_loss
            metric_str = f"{metric_val:.3f}" if metric_val is not None else "NA"
            name = f"checkpoint_{joined_txt}_e{self.metrics.epoch}_{'acc' if not joined else 'loss'}{metric_str}"
            self.checkpoint(name=name)


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

            logits = self._forward(X)

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
            self.save_regular_checkpoint(joined=False)

        # ---------------- save the final checkpoint ----------------
        if self.save_final_checkpoint:
            name = f"final_checkpoint_e{self.metrics.epoch}_acc{self.metrics.val_accuracy:.3f}"
            self.checkpoint(name=name)

        # ---------------- train after stopping ----------------
        if self.train_after_stopping:
            self.optimizer = self.optimizer_factory(self.model)
            self.scheduler = self.scheduler_factory(self.optimizer)
            logger.info(
                f"Training after stopping for {self.train_after_stopping_epochs} epochs"
            )
            self.metrics.val_accuracy = None  # set to None to avoid confusion
            self.metrics.val_loss = None  # set to None to avoid confusion
            self.patience_counter = 0
            for epoch in range(self.train_after_stopping_epochs):
                self.metrics.epoch += 1
                self.train_epoch(self.train_after_stopping_set)
                self.log_metrics()

                if (
                    self.metrics.train_loss <= self.best_val_loss
                ):  # use train_loss because val_loss is None
                    logger.info(
                        f"Training after stopping stopped at epoch {epoch} because target loss was reached"
                    )
                    break
                self.save_regular_checkpoint(joined=True)

            self.checkpoint(name=f"FINAL_e{self.metrics.epoch}")

    def check_early_stopping(self) -> bool:
        """Check if early stopping criteria is met. Returns True if should stop."""

        is_better = self.metrics.val_loss < (
            self.best_val_loss - self.early_stopping_delta
        )

        if is_better:
            self.best_val_accuracy = self.metrics.val_accuracy
            self.best_val_loss = self.metrics.val_loss
            self.best_val_epoch = self.metrics.epoch
            self.patience_counter = 0
            logger.info(
                f"New best val_loss: {self.best_val_loss:.4f}, val_accuracy: {self.best_val_accuracy:.4f}, patience: {self.patience_counter}/{self.early_stopping_patience}"
            )

            # Save best model
            if self.save_best_checkpoints:
                self.checkpoint(name="best_val_checkpoint")

        else:
            self.patience_counter += 1
            logger.info(
                f"Early stopping patience: {self.patience_counter}/{self.early_stopping_patience}"
            )

        if self.patience_counter >= self.early_stopping_patience:
            logger.info(
                f"Early stopping triggered! Best val_loss: {self.best_val_loss:.4f}, val_accuracy: {self.best_val_accuracy:.4f}"
            )
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

            logits = self._forward(X)

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

    def finish(self):
        # put all metrics to json and save it

        all_metrics = {k: v for k, v in self.metrics.__dict__.items() if v is not None}
        all_metrics["best_val_accuracy"] = self.best_val_accuracy
        all_metrics["best_val_loss"] = self.best_val_loss
        all_metrics["best_val_epoch"] = self.best_val_epoch
        all_metrics["patience_counter"] = self.patience_counter

        with open(self.checkpoint_root / "final_metrics.json", "w") as f:
            json.dump(all_metrics, f)

        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
