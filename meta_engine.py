import json
import logging
import time
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.func import functional_call

from meta_helpers import (
    build_episode_index,
    sample_support,
    sample_query,
    fetch_by_indices,
    sample_support_no_run,
    sample_query_no_run,
)


from torch.utils.data import DataLoader

from base_engine import BaseEngine
logger = logging.getLogger(__name__)


class Metrics:
    epoch: Optional[int] = None
    support_loss: Optional[float] = None
    query_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    scheduler_lr: Optional[float] = None
    grad_norm: Optional[float] = None
    train_loss_supervised: Optional[float] = None
    train_accuracy_supervised: Optional[float] = None
    val_loss_supervised: Optional[float] = None
    val_accuracy_supervised: Optional[float] = None
    scheduler_lr_supervised: Optional[float] = None


class MetaEngine(BaseEngine):
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
        meta_iterations: int,
        validate_every: int,
        validate_meta_every: int,
        # datasets / splits 
        train_ds,
        val_ds,
        test_ds,
        S_train: List[int],
        S_val: List[int],
        S_test: List[int],
        # loss / factories
        loss_fn: Optional[nn.Module],
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
        scheduler_factory: Callable[[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]],
        # episode design
        meta_batch_size: int,
        k_support: int,
        q_query: Optional[int],
        q_eval: Optional[int],
        val_episodes_per_subject: Optional[int],
        inner_steps: int,
        inner_lr: float,
        run_size: int,
        # RNG
        seed: int = 111,
        clip_grad_norm: float = None,
        # supervised-in-meta knobs
        n_epochs_supervised: int = 100,
        meta_iters_per_meta_epoch: int = 100,
        supervised_train_batch_size: int = 250,
        supervised_eval_batch_size: int = 250,
        supervised_optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
        supervised_scheduler_factory: Optional[Callable[[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]] = None,
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

        self.meta_iterations = int(meta_iterations)
        self.validate_every = validate_every
        self.validate_meta_every = validate_meta_every

        # datasets / splits
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.S_train = S_train
        self.S_val = S_val
        self.S_test = S_test

        # loss / factories
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        # episode design
        self.T = meta_batch_size
        self.K = k_support
        self.Q = q_query
        self.Q_eval = q_eval
        self.val_episodes_per_subject = val_episodes_per_subject
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

        # episode indices (explicit run_size)
        self.train_epi = build_episode_index(self.train_ds, run_size=run_size)
        self.val_epi = build_episode_index(self.val_ds, run_size=run_size)
        self.test_epi = build_episode_index(self.test_ds, run_size=run_size)

        # Seed
        self.rng = random.Random(int(seed))
        self.seed = seed
        self.clip_grad_norm = clip_grad_norm

        self._allow_trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        allow_params = [p for n, p in self.model.named_parameters() if n in self._allow_trainable]
        assert allow_params, "No allowed trainable params — check policy/target_modules."

        # opt/scheduler
        self.optimizer = optimizer_factory(allow_params)
        self.scheduler = scheduler_factory(self.optimizer)

        # supervised-in-meta knobs
        self.n_epochs_supervised = n_epochs_supervised
        self.meta_iters_per_meta_epoch = meta_iters_per_meta_epoch
        self.supervised_train_batch_size = supervised_train_batch_size
        self.supervised_eval_batch_size = supervised_eval_batch_size
        self.sup_optimizer = supervised_optimizer_factory(allow_params)
        self.sup_scheduler = supervised_scheduler_factory(self.sup_optimizer)

        # metrics
        self.metrics = Metrics()
        

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
