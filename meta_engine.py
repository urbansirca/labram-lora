import json
import logging
import time
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.func import functional_call
from models import load_labram
from preprocess_KU_data import get_ku_dataset_channels
from meta_helpers import (
    build_episode_index,
    sample_support,
    sample_query,
    fetch_by_indices,
)

logger = logging.getLogger(__name__)


class Metrics:
    epoch: Optional[int] = None
    # train-side (per outer step / epoch)
    support_loss: Optional[float] = None
    query_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    # val-side
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    # test-side
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    # optimizer/scheduler
    scheduler_lr: Optional[float] = None
    grad_norm: Optional[float] = None


class MetaEngine:
    def __init__(
        self,
        # --- core run identity/infra (mirrors Engine) ---
        model: nn.Module,
        model_str: str,
        experiment_name: str,
        device: Union[str, torch.device],
        meta_iterations: int,
        validate_every: int,
        # --- datasets / splits ---
        train_ds,
        val_ds,
        test_ds,
        S_train: List[int],
        S_val: List[int],
        S_test: List[int],
        # --- optim / sched (factories, like Engine) ---
        loss_fn: Optional[nn.Module],
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
        scheduler_factory: Callable[
            [torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]
        ],
        # --- episode design (explicit knobs) ---
        meta_batch_size: int,
        k_support: int,
        q_query: Optional[int],  # None => all remaining
        inner_steps: int,
        inner_lr: float,
        run_size: int,  # used for episode indexing
        # --- perf knobs (same names/semantics as Engine) ---
        use_amp: bool = True,
        non_blocking: bool = True,
        pin_memory: bool = False,
        use_compile: bool = False,
        # --- logging (same knobs) ---
        use_wandb: bool = False,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        config_for_logging: Optional[Dict] = None,  # optional, just to dump
        # --- checkpoints (same knobs) ---
        save_regular_checkpoints: bool = False,
        save_final_checkpoint: bool = True,
        save_regular_checkpoints_interval: int = 10,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        # --- RNG ---
        seed: int = 111,
        # --- Labram ---
        n_patches_labram: int = 4,
        samples: int = 200,
        channels: int = 62,
        electrodes: List[str] = None,
        clip_grad_norm = None,
        q_eval = None,
    ):
        # device / model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        if use_compile:
            self.model = torch.compile(self.model)
        self.model_str = model_str
        self.use_amp = use_amp

        # id / meta_iterations
        self.experiment_name = experiment_name
        self.meta_iterations = int(meta_iterations)

        # datasets / splits
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds
        self.S_train, self.S_val, self.S_test = list(S_train), list(S_val), list(S_test)

        # episode knobs
        self.T = int(meta_batch_size)
        self.K = int(k_support)
        self.Q = q_query
        self.Q_eval = q_eval or q_query
        self.inner_steps = int(inner_steps)
        self.inner_lr = float(inner_lr)
        self.validate_every = int(validate_every)

        # episode indices (explicit run_size)
        self.train_epi = build_episode_index(self.train_ds, run_size=run_size)
        self.val_epi = build_episode_index(self.val_ds, run_size=run_size)
        self.test_epi = build_episode_index(self.test_ds, run_size=run_size)

        # optim / sched
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.clip_grad_norm = clip_grad_norm

        # perf
        self.non_blocking = non_blocking
        self.pin_memory = pin_memory

        # logging
        self.use_wandb = bool(use_wandb)
        self.wandb_run = None
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.config_for_logging = config_for_logging

        # Labram
        self.n_patches_labram = n_patches_labram
        self.samples = samples
        self.channels = channels
        self.electrodes = electrodes

        if self.use_wandb:
            import wandb

            assert self.wandb_project, "wandb_project must be set when use_wandb=True"
            logger.info(
                f"Setting up wandb: {self.wandb_entity}/{self.wandb_project} :: {self.experiment_name}"
            )
            self.wandb_run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=self.experiment_name,
                config=self.config_for_logging,
            )

        # checkpoints
        self.save_regular_checkpoints = save_regular_checkpoints
        self.save_final_checkpoint = save_final_checkpoint
        self.save_regular_checkpoints_interval = int(save_regular_checkpoints_interval)
        self.checkpoint_root = (
            Path(checkpoint_dir)
            if checkpoint_dir
            else (
                Path(__file__).parent / "weights" / "checkpoints_meta" / self.experiment_name
            )
        )
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

        # dump config snapshot (optional)
        if self.config_for_logging is not None:
            with open(self.checkpoint_root / "config.json", "w") as f:
                json.dump(self.config_for_logging, f, indent=2)

        # misc
        self.metrics = Metrics()
        self._t0 = time.time()
        self._printed_header = False
        self.rng = random.Random(int(seed))

        self.base_param_shapes = [p.shape for p in self.model.parameters()]

        self._audit_trainables()
        self._allow_trainable = self._expected_trainable_names()

        allow_params = [
            p for n, p in self.model.named_parameters() if n in self._allow_trainable
        ]
        assert (
            allow_params
        ), "No allowed trainable params — check policy/target_modules."
        self.optimizer = optimizer_factory(allow_params)
        self.scheduler = scheduler_factory(self.optimizer)

    def _expected_trainable_names(self):
        names = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                names.append(n)
        return set(names)

    def _audit_trainables(self):
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

        trainables = [
            (n, p.numel()) for n, p in self.model.named_parameters() if p.requires_grad
        ]
        frozen = [
            (n, p.numel())
            for n, p in self.model.named_parameters()
            if not p.requires_grad
        ]
        n_tr = sum(k for _, k in trainables)
        n_fr = sum(k for _, k in frozen)
        logger.info(f"Trainable params: {len(trainables)} tensors, {n_tr} elems")
        logger.info(f"Frozen params:    {len(frozen)} tensors, {n_fr} elems")

    def _grad_norm(self, params) -> float:
        grad_tensors = [p.grad.detach() for p in params if p.grad is not None]
        if not grad_tensors:
            return 0.0
        return torch.norm(torch.stack([torch.norm(g) for g in grad_tensors])).item()

    def _clone_as_leaf(self, params):
        return [
            torch.empty_like(p).copy_(p.detach()).requires_grad_(True) for p in params
        ]

    def _sgd_update_detached(self, fast, grads, lr):
        out = []
        for p, g in zip(fast, grads):
            if g is None:
                out.append(p)
            else:
                out.append((p - lr * g).detach().requires_grad_(True))
        return out

    def _forward_with(self, params_dict, x):
        if self.use_amp:
            with torch.autocast(device_type=self.device.type):
                return functional_call(
                    self.model,
                    params_dict,
                    args=(),
                    kwargs={"x": x, "electrodes": self.electrodes},
                )
        else:
            return functional_call(
                self.model,
                params_dict,
                args=(),
                kwargs={"x": x, "electrodes": self.electrodes},
            )

    def _forward(self, X):
        if self.use_amp:
            with torch.autocast(device_type=self.device.type):
                return self.model(x=X, electrodes=self.electrodes)
        else:
            return self.model(x=X, electrodes=self.electrodes)

    # ---------- logging (compact console; wandb optional) ----------
    def log_metrics(self):
        m = {k: v for k, v in self.metrics.__dict__.items() if v is not None}
        line = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in m.items()
        )
        line += f" | Runtime: {time.time() - self._t0:.2f}s"
        logger.info(line)
        self._t0 = time.time()
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log(m)

    # ---------- checkpoints (same behavior as Engine) ----------
    def checkpoint(self, name: str = "checkpoint"):
        out_stem = str(self.checkpoint_root / name)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out_stem)
        else:
            torch.save(self.model.state_dict(), f"{out_stem}.pt")
        metadata = {
            "iteration": getattr(self.metrics, "iteration", None),
            "train_accuracy": getattr(self.metrics, "train_accuracy", None),
            "val_accuracy": getattr(self.metrics, "val_accuracy", None),
            "query_loss": getattr(self.metrics, "query_loss", None),
            "val_loss": getattr(self.metrics, "val_loss", None),
            "scheduler_lr": getattr(self.metrics, "scheduler_lr", None),
            "timestamp": time.time(),
        }
        with open(f"{out_stem}_training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_regular_checkpoint(self):
        if (
            self.save_regular_checkpoints
            and (self.metrics.iteration % self.save_regular_checkpoints_interval == 0) 
        ):
            metric_val = (
                self.metrics.val_accuracy
                if self.metrics.val_accuracy is not None
                else self.metrics.train_accuracy
            )
            metric_str = f"{metric_val:.3f}" if metric_val is not None else "NA"
            name = f"checkpoint_i{self.metrics.iteration}_acc{metric_str}"
            self.checkpoint(name=name)

    def meta_step(self, subjects_batch: List[int]):
        self.model.eval()  # freeze BN running stats during meta-episode
        base_named = [
            (n, p)
            for n, p in self.model.named_parameters()
            if n in self._allow_trainable
        ]
        base_names = [n for n, _ in base_named]
        base_params = [p for _, p in base_named]

        self.optimizer.zero_grad(set_to_none=True)

        outer_losses = []
        inner_losses = []

        correct_preds = []
        total_samples = []

        for sid in subjects_batch:
            sup_idx, que_runs = sample_support(sid, self.train_epi, self.K, self.rng)
            que_idx = sample_query(sid, que_runs, self.train_epi, self.Q, self.rng)
            Xs, ys = fetch_by_indices(
                self.train_ds,
                self.train_epi,
                sid,
                sup_idx,
                self.device,
                self.non_blocking,
            )
            Xq, yq = fetch_by_indices(
                self.train_ds,
                self.train_epi,
                sid,
                que_idx,
                self.device,
                self.non_blocking,
            )

            fast = self._clone_as_leaf(base_params)
            fast_dict = dict(zip(base_names, fast))
            for _ in range(self.inner_steps):
                for name, param in zip(base_names, fast):
                    fast_dict[name] = param
                logits_s = self._forward_with(fast_dict, Xs)
                Ls = nn.functional.cross_entropy(logits_s, ys)
                inner_losses.append(Ls.detach())
                grads = torch.autograd.grad(
                    Ls, fast, create_graph=False, retain_graph=False, allow_unused=True
                )
                fast = self._sgd_update_detached(fast, grads, self.inner_lr)

            for name, param in zip(base_names, fast):
                fast_dict[name] = param
            logits_q = self._forward_with(fast_dict, Xq)
            Lq = nn.functional.cross_entropy(logits_q, yq)

            grads_q = torch.autograd.grad(
                Lq, fast, retain_graph=False, allow_unused=True
            )
            scale = 1.0 / max(1, len(subjects_batch))
            for bp, gq in zip(base_params, grads_q):
                if gq is None:
                    continue
                g = gq.detach() * scale
                if bp.grad is None:
                    bp.grad = g.clone()
                else:
                    bp.grad.add_(g)

            outer_losses.append(Lq.detach())
            correct_preds.append((logits_q.argmax(1) == yq).sum())
            total_samples.append(torch.tensor(yq.numel(), device=self.device))

        if self.clip_grad_norm:
            total_norm = torch.nn.utils.clip_grad_norm_(base_params, max_norm=self.clip_grad_norm)
            self.metrics.grad_norm = float(total_norm)  
        else:
            self.metrics.grad_norm = self._grad_norm(base_params)
        self.optimizer.step()

        outer_correct = torch.stack(correct_preds).sum().item()
        outer_count = torch.stack(total_samples).sum().item()
        self.metrics.support_loss = (
            torch.stack(inner_losses).mean().cpu().item() if inner_losses else None
        )
        self.metrics.query_loss = (
            torch.stack(outer_losses).mean().cpu().item() if outer_losses else None
        )
        self.metrics.train_accuracy = outer_correct / max(1, outer_count)
        self.metrics.scheduler_lr = float(self.optimizer.param_groups[0]["lr"])
        self.metrics.grad_norm = self._grad_norm(base_params)

    def meta_validate_epoch(self):
        """Adapt per subject on K shots, measure query. No outer update, no model mutation."""
        self.model.eval()
        rng = self.rng

        total_losses = []
        total_corrects = []
        total_count = 0

        # we never mutate base params; we only read them to create 'fast'
        base_named = [
            (n, p)
            for n, p in self.model.named_parameters()
            if n in self._allow_trainable
        ]
        base_names = [n for n, _ in base_named]
        base_params = [p for _, p in base_named]

        for sid in self.S_val:
            sup_idx, que_runs = sample_support(sid, self.val_epi, self.K, rng)
            que_idx = sample_query(sid, que_runs, self.val_epi, self.Q_eval, rng)

            Xs, ys = fetch_by_indices(
                self.val_ds,
                self.val_epi,
                sid,
                sup_idx,
                self.device,
                self.non_blocking,
            )
            Xq, yq = fetch_by_indices(
                self.val_ds,
                self.val_epi,
                sid,
                que_idx,
                self.device,
                self.non_blocking,
            )

            # ----- inner adaptation on cloned leafs (requires grad) -----
            fast = self._clone_as_leaf(base_params)  # leaves w/ requires_grad=True
            fast_dict = dict(zip(base_names, fast))
            for _ in range(self.inner_steps):
                for name, param in zip(base_names, fast):
                    fast_dict[name] = param
                # enable grad just for inner step
                with torch.enable_grad():
                    logits_s = self._forward_with(fast_dict, Xs)
                    Ls = torch.nn.functional.cross_entropy(logits_s, ys)
                    grads = torch.autograd.grad(
                        Ls,
                        fast,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True,
                    )
                fast = self._sgd_update_detached(fast, grads, self.inner_lr)

            # ----- query evaluation with adapted params (no grad needed) -----
            for name, param in zip(base_names, fast):
                fast_dict[name] = param
            with torch.no_grad():
                logits_q = self._forward_with(fast_dict, Xq)
                Lq = torch.nn.functional.cross_entropy(logits_q, yq)
                total_losses.append(Lq)
                total_corrects.append((logits_q.argmax(1) == yq).sum())
                total_count += yq.numel()

        total_correct = torch.stack(total_corrects).sum().item()

        self.metrics.val_loss = torch.stack(total_losses).mean().cpu().item()
        self.metrics.val_accuracy = float(total_correct) / max(1, int(total_count))

    def train(self):
        for i in range(1, self.meta_iterations + 1):
            self.metrics.iteration = i
            T = min(self.T, len(self.S_train))
            subjects_batch = self.rng.sample(self.S_train, k=T)
            self.meta_step(subjects_batch)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.validate_every > 0 and (i % self.validate_every) == 0:
                self.meta_validate_epoch()
                self.log_metrics()
                self.save_regular_checkpoint()

        if self.save_final_checkpoint:
            name = f"final_checkpoint_i{self.metrics.iteration}_acc{self.metrics.val_accuracy:.3f}"
            self.checkpoint(name=name)