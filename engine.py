import logging

import wandb
import torch.nn as nn


logger = logging.getLogger(__name__)


class Metrics:
    epoch = None
    train_loss = None
    val_loss = None
    train_accuracy = None
    val_accuracy = None
    support_loss = None
    query_loss = None
    scheduler_lr = None


class Engine:

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        hyperparameters: dict,
        experiment_name: str,
        n_epochs: int,
        device: str,
        batch_size: int,
        training_set,
        validation_set,
        test_set,
    ):
        self.model = model
        self.hyperparameters = hyperparameters
        self.experiment_name = experiment_name
        self.n_epochs = n_epochs
        self.device = device
        self.batch_size = batch_size
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.wandb_run = self.wandb_setup(config)
        self.metrics = Metrics()

    def wandb_setup(self, config):
        return wandb.init(
            entity="urban-sirca-vrije-universiteit-amsterdam",
            project="EEG-FM",
            name=self.experiment_name,
            config=config,
        )

    def wandb_log(self):
        """
        Log metrics to wandb. Values that are not available aren't logged.
        """
        present_metrics = {
            k: v for k, v in self.metrics.__dict__.items() if v is not None
        }
        self.wandb_run.log(present_metrics)

    def console_log(self):
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

    def train(self):
        for epoch in range(self.n_epochs):
            self.train_epoch()
            self.validate_epoch()
            self.wandb_log()
            self.console_log()

    def test(self):
        pass

    def finish(self):
        self.wandb_run.finish()
