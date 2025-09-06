import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import wandb

logger = logging.getLogger(__name__)


class Metrics:
    epoch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
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
    ):
        self.model = model
        self.hyperparameters = hyperparameters
        self.experiment_name = experiment_name
        self.n_epochs = n_epochs
        self.device = device
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.wandb_run = self.wandb_setup(config)
        self.metrics = Metrics()

    def wandb_setup(self, config: Dict[str, Any]) -> wandb.sdk.wandb_run.Run:
        return wandb.init(
            entity="urban-sirca-vrije-universiteit-amsterdam",
            project="EEG-FM",
            name=self.experiment_name,
            config=config,
        )

    def wandb_log(self) -> None:
        """
        Log metrics to wandb. Values that are not available aren't logged.
        """
        present_metrics = {
            k: v for k, v in self.metrics.__dict__.items() if v is not None
        }
        self.wandb_run.log(present_metrics)

    def console_log(self) -> None:
        """
        Log metrics to console. Values that are not available aren't logged.
        """
        present_metrics = {
            k: v for k, v in self.metrics.__dict__.items() if v is not None
        }
        present_metrics["epoch"] = f"{self.metrics.epoch:2d}/{self.n_epochs}"
        present_metrics = " | ".join(
            [f"{k}: {v:.4f}" for k, v in present_metrics.items()]
        )
        logger.info(present_metrics)

    def train(self) -> None:
        for epoch in range(self.n_epochs):
            self.train_epoch()
            self.validate_epoch()
            self.wandb_log()
            self.console_log()

    def test(self) -> None:
        pass

    def finish(self) -> None:
        self.wandb_run.finish()
