import json, logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import torch
import torch.nn as nn
import time
logger = logging.getLogger(__name__)

class BaseEngine:
    def __init__(
        self,
        model_str: str,
        experiment_name: str,
        device: Union[str, torch.device],
        model: nn.Module,
        electrodes: Optional[List[str]],
        # misc
        non_blocking: bool,
        pin_memory: bool,
        use_amp: bool,
        use_compile: bool,
        # wandb/logging
        use_wandb: bool,
        wandb_entity: Optional[str],
        wandb_project: Optional[str],
        config_for_logging: Optional[Dict[str, Any]],
        # checkpoints
        save_regular_checkpoints: bool,
        save_regular_checkpoints_interval: int,
        save_best_checkpoints: bool,
        save_final_checkpoint: bool,
        checkpoint_dir: Optional[Union[str, Path]],
    ):
        self.model_str = model_str
        self.experiment_name = experiment_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.electrodes = electrodes

        self.non_blocking = non_blocking
        self.pin_memory = pin_memory
        self.use_amp = use_amp
        if use_compile:
            self.model = torch.compile(self.model)

        # wandb
        self.use_wandb = bool(use_wandb)
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.config_for_logging = config_for_logging
        self.wandb_run = None
        if self.use_wandb:
            import wandb
            assert self.wandb_project, "wandb_project must be set when use_wandb=True"
            logger.info(f"Setting up wandb: {self.wandb_entity}/{self.wandb_project} :: {self.experiment_name}")
            self.wandb_run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.experiment_name,
                config=self.config_for_logging,
            )

        # checkpoints
        self.save_regular_checkpoints = save_regular_checkpoints
        self.save_regular_checkpoints_interval = int(save_regular_checkpoints_interval)
        self.save_best_checkpoints = save_best_checkpoints
        self.save_final_checkpoint = save_final_checkpoint
        self.checkpoint_root = Path(checkpoint_dir) # if checkpoint_dir else (Path(__file__).parent / "weights" / "checkpoints" / self.experiment_name)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        
        self.runtime_list = []

        # dump config for reproducibility
        if self.config_for_logging is not None:
            with open(self.checkpoint_root / "config.json", "w") as f:
                json.dump(self.config_for_logging, f, indent=2)

        self._audit_trainables()
        
        self.start_time = time.time()

    
    def _audit_trainables(self):
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        named_params = list(self.model.named_parameters())
        trainables = [(n, p.numel()) for n, p in named_params if p.requires_grad]
        frozen = [(n, p.numel()) for n, p in named_params if not p.requires_grad]
        n_tr = sum(k for _, k in trainables)
        n_fr = sum(k for _, k in frozen)
        logger.info(f"Trainable params: {len(trainables)} tensors, {n_tr} elems")
        logger.info(f"Frozen params:    {len(frozen)} tensors, {n_fr} elems")


    def log_metrics(self):
        present = {k: v for k, v in self.metrics.__dict__.items() if v is not None}
        line = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in present.items()
        )
        line += f" | Runtime: {time.time() - self.start_time:.2f}s"
        logger.info(line)
        self.runtime_list.append(time.time() - self.start_time)
        self.start_time = time.time()
        # wandb
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log(present)

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.use_amp:
            with torch.autocast(device_type=self.device.type):
                return self.model(x=X, electrodes=self.electrodes)
        else:
            return self.model(x=X, electrodes=self.electrodes)
        
    def checkpoint(self, name: str = "checkpoint"):
        out_stem = str(self.checkpoint_root / name)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out_stem)
        else:
            torch.save(self.model.state_dict(), f"{out_stem}.pt")

        metadata = {k: v for k, v in self.metrics.__dict__.items() if v is not None}
        metadata.update({
            "timestamp": time.time(),
            "experiment_name": self.experiment_name,
            "model_str": self.model_str,
        })
        
        with open(f"{out_stem}_training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)