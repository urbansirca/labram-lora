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
        model_str: str,
        experiment_name: str,
        device: str,
        n_epochs: int,

        # data loaders
        training_set: DataLoader,
        validation_set: DataLoader,
        test_set: Optional[DataLoader],
        train_after_stopping_set: Optional[DataLoader],

        # optim / sched
        # optimizer: Optimizer,
        # scheduler: Optional[_LRScheduler],
        loss_fn: Optional[nn.Module],

        # factories for optim / sched
        optimizer_factory: Callable[[nn.Module], Optimizer],
        scheduler_factory: Callable[[Optimizer], Optional[_LRScheduler]],

        # data shape (explicit, used by assert_dimensions)
        input_channels: int,
        trial_length: int,
        n_patches_labram: Optional[int] = None,
        patch_length: Optional[int] = None,

        # model extras
        electrodes: Optional[List[str]] = None,

        # perf knobs
        use_compile: bool = False,
        non_blocking: bool = True,
        pin_memory: bool = False,
        use_amp: bool = True,

        # logging
        use_wandb: bool = False,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,

        # checkpoints
        save_regular_checkpoints: bool = False,
        save_final_checkpoint: bool = True,
        save_best_checkpoints: bool = True,
        save_regular_checkpoints_interval: int = 10,
        checkpoint_dir: Optional[Union[str, Path]] = None,

        # early stopping
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_delta: float = 0.0,

        # train-after-stopping
        train_after_stopping: bool = False,
        train_after_stopping_epochs: int = 0,

        # --- purely for logging/serialization (never read for logic) ---
        config_for_logging: Optional[Dict[str, Any]] = None,
    ):
        self.model_str = model_str
        self.experiment_name = experiment_name
        self.n_epochs = int(n_epochs)
        self.device = torch.device(device) if isinstance(device, str) else device

        # perf
        self.non_blocking = non_blocking
        self.pin_memory = pin_memory
        self.use_amp = use_amp

        # model
        self.model = model.to(self.device)
        if use_compile:
            self.model = torch.compile(self.model)

        # loaders
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.train_after_stopping_set = train_after_stopping_set

        # opt/sched
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.optimizer = self.optimizer_factory(self.model)
        self.scheduler = self.scheduler_factory(self.optimizer)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        # shapes
        self.input_channels = input_channels
        self.trial_length = trial_length
        self.n_patches_labram = n_patches_labram
        self.patch_length = patch_length

        # Labram 
        self.electrodes = electrodes

        self.config_for_logging = config_for_logging

        # wandb
        self.use_wandb = bool(use_wandb) and (wandb is not None)
        self.wandb_run = None
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        
        if self.use_wandb:
            assert self.wandb_project, "wandb_project must be set when use_wandb=True"
            self.wandb_run = self.wandb_setup()


        # checkpoints
        self.save_regular_checkpoints = save_regular_checkpoints
        self.save_final_checkpoint = save_final_checkpoint
        self.save_best_checkpoints = save_best_checkpoints
        self.save_regular_checkpoints_interval = int(save_regular_checkpoints_interval)
        self.checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else (Path(__file__).parent / "weights" / "checkpoints" / self.experiment_name)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

        # dump config for reproducibility
        if self.config_for_logging is not None:
            with open(self.checkpoint_root / "config.json", "w") as f:
                json.dump(self.config_for_logging, f, indent=2)

        # metrics/ES
        self.metrics = Metrics()
        self.best_val_accuracy = 0.0
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0
        self.patience_counter = 0
        self.early_stopping = early_stopping
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_delta = float(early_stopping_delta)

        # train-after-stopping
        self.train_after_stopping = train_after_stopping
        self.train_after_stopping_epochs = int(train_after_stopping_epochs)

        self.start_time = time.time()

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

    def setup_optimizations(self):
        torch.backends.cudnn.benchmark = True

        # Optional: Set memory format for better performance on modern GPUs
        # (Only if your model supports channels-last)
        # torch.backends.cudnn.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_tf32 = True

    def checkpoint(self, name: str = "checkpoint"):
        out_stem = str(self.checkpoint_root / str(name))

        # Save depending on model type
        if hasattr(self.model, "save_pretrained"):  
            # e.g. PEFT/LoRA/HF-style
            self.model.save_pretrained(out_stem)
        else:
            # Plain PyTorch nn.Module
            torch.save(self.model.state_dict(), f"{out_stem}.pt")

        # Save training metadata
        metadata = {
            "epoch": getattr(self.metrics, "epoch", None),
            "train_loss": getattr(self.metrics, "train_loss", None),
            "val_loss": getattr(self.metrics, "val_loss", None),
            "train_accuracy": getattr(self.metrics, "train_accuracy", None),
            "val_accuracy": getattr(self.metrics, "val_accuracy", None),
            "scheduler_lr": getattr(self.metrics, "scheduler_lr", None),
            "timestamp": time.time(),
        }
        with open(f"{out_stem}_training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_regular_checkpoint(self, joined: bool = False):
        if self.save_regular_checkpoints and self.metrics.epoch % self.save_regular_checkpoints_interval == 0:
            joined_txt = "JOINED" if joined else ""
            metric_val = self.metrics.val_accuracy if not joined else self.metrics.train_loss
            metric_str = f"{metric_val:.3f}" if metric_val is not None else "NA"
            name = f"checkpoint_{joined_txt}_e{self.metrics.epoch}_{'acc' if not joined else 'loss'}{metric_str}"
            self.checkpoint(name=name)

    def wandb_setup(self):
        logger.info(f"Setting up wandb: {self.wandb_entity}/{self.wandb_project} :: {self.experiment_name}")
        return wandb.init(
            entity=self.wandb_entity,
            project=self.wandb_project,
            name=self.experiment_name,
            config=self.config_for_logging,
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

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Unified forward pass with optional autocast and model_str dispatch.
        """
        if self.use_amp:
            with torch.autocast(device_type=self.device.type):
                return self.model(x=X, electrodes=self.electrodes) if self.model_str == "labram" else self.model(X)
        else:
            return self.model(x=X, electrodes=self.electrodes) if self.model_str == "labram" else self.model(X)


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

        self.metrics.test_loss = (total_loss / max(1, total_count)).item()
        self.metrics.test_accuracy = (total_correct / max(1, total_count)).item()

        # log once for visibility
        self.log_metrics()

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
