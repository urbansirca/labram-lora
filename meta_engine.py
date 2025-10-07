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
    sample_support_no_run,
    sample_query_no_run,
)

from torch.utils.data import DataLoader


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

    train_loss_supervised: Optional[float] = None
    train_accuracy_supervised: Optional[float] = None
    val_loss_supervised: Optional[float] = None
    val_accuracy_supervised: Optional[float] = None
    scheduler_lr_supervised: Optional[float] = None


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
        validate_meta_every: int,
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
        # meta_batch_size: int,
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
        save_regular_checkpoints_interval: int = 10,
        save_best_checkpoints: bool = True,
        save_final_checkpoint: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        # --- RNG ---
        seed: int = 111,
        # --- Labram ---
        channels: int = 62,
        electrodes: List[str] = None,
        clip_grad_norm: float = None,
        q_eval: int = None,
        val_episodes_per_subject: int = None,
        n_epochs_supervised: int = 100,
        meta_iters_per_meta_epoch: int = 100,
        supervised_train_batch_size: int = 250,
        supervised_eval_batch_size: int = 250,
        # --- extra (supervised) factories; default to the meta factories ---
        supervised_optimizer_factory: Optional[
            Callable[[nn.Module], torch.optim.Optimizer]
        ] = None,
        supervised_scheduler_factory: Optional[
            Callable[
                [torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]
            ]
        ] = None,
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
        self.K = int(k_support)
        self.Q = q_query
        self.Q_eval = q_eval or q_query
        self.inner_steps = int(inner_steps)
        self.inner_lr = float(inner_lr)
        self.validate_every = int(validate_every)
        self.validate_meta_every = int(validate_meta_every)

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
        self.save_best_checkpoints = save_best_checkpoints
        self.save_regular_checkpoints = save_regular_checkpoints
        self.save_final_checkpoint = save_final_checkpoint
        self.save_regular_checkpoints_interval = int(save_regular_checkpoints_interval)
        self.checkpoint_root = (
            Path(checkpoint_dir)
            if checkpoint_dir
            else (
                Path(__file__).parent
                / "weights"
                / "checkpoints_meta"
                / self.experiment_name
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
        self.seed = seed

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

        self.val_episodes_per_subject = val_episodes_per_subject

        # -------- supervised (normal) optimizer/scheduler --------
        self.sup_optimizer_factory = (
            supervised_optimizer_factory  # or optimizer_factory
        )
        self.sup_scheduler_factory = (
            supervised_scheduler_factory  # or scheduler_factory
        )

        # we’ll train the same allowed params by default; override policy here if needed
        sup_params = [
            p for n, p in self.model.named_parameters() if n in self._allow_trainable
        ]
        self.sup_optimizer = self.sup_optimizer_factory(sup_params)
        self.sup_scheduler = self.sup_scheduler_factory(self.sup_optimizer)

        self.n_epochs_supervised = n_epochs_supervised
        self.meta_iters_per_meta_epoch = meta_iters_per_meta_epoch
        self.supervised_train_batch_size = supervised_train_batch_size
        self.supervised_eval_batch_size = supervised_eval_batch_size

        self.sup_optimizer = self.sup_optimizer_factory(sup_params)
        self.sup_scheduler = self.sup_scheduler_factory(self.sup_optimizer)

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
            "train_accuracy_supervised": getattr(self.metrics, "train_accuracy_supervised", None),
            "val_accuracy_supervised": getattr(self.metrics, "val_accuracy_supervised", None),
            "scheduler_lr_supervised": getattr(self.metrics, "scheduler_lr_supervised", None),
            "train_loss_supervised": getattr(self.metrics, "train_loss_supervised", None),
            "val_loss_supervised": getattr(self.metrics, "val_loss_supervised", None),
            "timestamp": time.time(),
        }
        with open(f"{out_stem}_training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def save_regular_checkpoint(self):
        if self.save_regular_checkpoints and (
            self.metrics.iteration % self.save_regular_checkpoints_interval == 0
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
        self.model.eval()  # freeze BN running stats during meta-episode #TODO: check with professor
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
            sup_idx, que_runs = sample_support_no_run(
                sid, self.train_epi, self.K, self.rng
            )
            que_idx = sample_query_no_run(
                sid, que_runs, self.train_epi, self.Q, self.rng
            )
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
            total_norm = torch.nn.utils.clip_grad_norm_(
                base_params, max_norm=self.clip_grad_norm
            )
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
        self.model.eval() #TODO: check with professor
        rng = self.rng

        E = int(getattr(self, "val_episodes_per_subject", 0))

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
            for _ in range(max(1, E)):
                sup_idx, que_runs = sample_support(sid, self.val_epi, self.K, rng)
                que_idx = sample_query(sid, que_runs, self.val_epi, self.Q_eval, rng)
                # sup_idx, que_runs = sample_support_no_run(sid, self.val_epi, self.K, self.rng)
                # que_idx = sample_query_no_run(sid, que_runs, self.val_epi, self.Q_eval, self.rng)  

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

    def meta_train(self):
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

    def supervised_train_epoch(self, dataloader=None) -> None:
        """Standard supervised epoch using self.sup_optimizer / self.sup_scheduler."""
        if dataloader is None:
            if self.sup_train_loader is None:
                raise ValueError(
                    "No train dataloader provided. Call setup_supervised_dataloaders(...) first or pass a dataloader."
                )
            dataloader = self.sup_train_loader

        self.model.train()
        device = self.device
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0, device=device)
        total_count = torch.tensor(0, device=device)

        for X, Y, *_ in dataloader:
            X = X.to(device, non_blocking=self.non_blocking)
            Y = Y.long().to(device, non_blocking=self.non_blocking)

            self.sup_optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.autocast(device_type=device.type):
                    logits = self._forward(X)
                    loss = self.loss_fn(logits, Y)
            else:
                logits = self._forward(X)
                loss = self.loss_fn(logits, Y)

            loss.backward()

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    max_norm=self.clip_grad_norm,
                )

            self.sup_optimizer.step()

            bsz = Y.size(0)
            total_loss += loss * bsz
            total_correct += (logits.argmax(1) == Y).sum()
            total_count += bsz

        self.metrics.train_loss_supervised = (total_loss / max(1, total_count)).item()
        self.metrics.train_accuracy_supervised = (
            total_correct / max(1, total_count)
        ).item()
        self.metrics.scheduler_lr_supervised = float(
            self.sup_optimizer.param_groups[0]["lr"]
        )

        if self.sup_scheduler is not None:
            self.sup_scheduler.step()

    @torch.no_grad()
    def supervised_validate_epoch(self, dataloader=None) -> None:
        """Standard supervised validation (no adaptation)."""
        if dataloader is None:
            if self.sup_val_loader is None:
                raise ValueError(
                    "No val dataloader provided. Call setup_supervised_dataloaders(...) first or pass a dataloader."
                )
            dataloader = self.sup_val_loader

        self.model.eval()
        device = self.device
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0, device=device)
        total_count = torch.tensor(0, device=device)

        for X, Y, *_ in dataloader:
            X = X.to(device, non_blocking=self.non_blocking)
            Y = Y.long().to(device, non_blocking=self.non_blocking)

            if self.use_amp:
                with torch.autocast(device_type=device.type):
                    logits = self._forward(X)
                    loss = self.loss_fn(logits, Y)
            else:
                logits = self._forward(X)
                loss = self.loss_fn(logits, Y)

            bsz = Y.size(0)
            total_loss += loss * bsz
            total_correct += (logits.argmax(1) == Y).sum()
            total_count += bsz

        self.metrics.val_loss_supervised = (total_loss / max(1, total_count)).item()
        self.metrics.val_accuracy_supervised = (
            total_correct / max(1, total_count)
        ).item()

    def setup_supervised_dataloaders(self) -> None:
        """
        Create standard PyTorch DataLoaders from the meta engine's datasets for
        plain supervised training/validation/testing.
        """
        g = torch.Generator().manual_seed(
            self.seed if self.seed is not None else int(self.rng.random() * 1e9)
        )
        eval_bs = self.supervised_eval_batch_size or self.supervised_train_batch_size

        num_workers = 0
        persistent_workers = False

        self.sup_train_loader = DataLoader(
            self.train_ds,
            batch_size=self.supervised_train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            generator=g,
            persistent_workers=persistent_workers,
        )
        self.sup_val_loader = DataLoader(
            self.val_ds,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
        )

    def train_alternating(
        self,
    ) -> None:
        """
        Alternates: [1 supervised epoch] -> [N meta-iterations], for total_epochs cycles.

        Preconditions:
        - Call setup_supervised_dataloaders(...) once, or pass loaders into
            supervised_* methods yourself.

        Args:
        total_epochs: number of cycles (each cycle = 1 supervised epoch + meta block)
        meta_iters_per_meta_epoch: meta iterations per cycle
        validate_meta_every: 0 = only validate at end of meta block; K = every K iters
        checkpoint_supervised: if True, also checkpoint right after supervised val
        """

        self.setup_supervised_dataloaders()
        # keep a running meta-iteration index so save_regular_checkpoint() still works
        iter_idx = int(getattr(self.metrics, "iteration", 0) or 0)

        best_val_accuracy = 0

        for epoch in range(1, int(self.n_epochs_supervised) + 1):
            self.metrics.epoch = epoch

            # -------- 1) supervised epoch (normal loss) --------
            self.supervised_train_epoch()  # uses self.sup_train_loader
            self.supervised_validate_epoch()  # uses self.sup_val_loader
      
            # -------- 2) meta block (N iterations) -------------
            for j in range(1, int(self.meta_iters_per_meta_epoch) + 1):
                iter_idx += 1
                self.metrics.iteration = iter_idx
                self.meta_step(self.S_train)

                if self.scheduler is not None:
                    self.scheduler.step()

                # optional intra-block meta validation/logging/checkpointing
                if self.validate_meta_every > 0 and (j % self.validate_meta_every == 0):
                    self.meta_validate_epoch()
                    # self.log_metrics()

            # end-of-block meta validation if user asked for "only at meta epoch end"
            self.meta_validate_epoch()
            self.log_metrics()
            self.save_regular_checkpoint()


            if self.metrics.val_accuracy > best_val_accuracy and self.save_best_checkpoints:
                best_val_accuracy = self.metrics.val_accuracy
                self.checkpoint(name=f"best_val_checkpoint_i_acc{best_val_accuracy:.3f}")

        # final checkpoint naming mirrors your meta `train()`
        if self.save_final_checkpoint:
            acc = (
                self.metrics.val_accuracy
                if self.metrics.val_accuracy is not None
                else self.metrics.train_accuracy
            )
            acc_str = f"{acc:.3f}" if acc is not None else "NA"
            self.checkpoint(
                name=f"final_checkpoint_i{self.metrics.iteration}_acc{acc_str}"
            )
