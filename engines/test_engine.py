import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Tuple
import wandb
import random
import torch
import torch.nn as nn
from torch.func import functional_call
import pandas as pd
from engines.utils import (
    sample_support,
    sample_query,
    fetch_by_indices,
    build_episode_index,
)

from models import freeze_all_but_head_labram, freeze_all_but_head_deepconvnet

logger = logging.getLogger(__name__)


class TestEngine:
    def __init__(
        self,
        engine,
        test_ds,
        use_wandb: bool = False,
        wandb_prefix: str = "test",
        run_size: int = 100,
        save_dir: Union[str, Path] = None,
        head_only: bool = False,
        test_lr: float = None,
        test_wd: float = None,
    ):
        self.engine = engine
        self.optimizer_factory = engine.optimizer_factory

        self.rng = random.Random(int(111))  # TODO: get it from engine maybe
        self.S_test = test_ds.subject_ids
        self.test_ds = test_ds
        self.test_epi = build_episode_index(self.test_ds, run_size=run_size)

        self.use_wandb = use_wandb
        self.wandb_prefix = wandb_prefix
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        
        
        self.head_only = head_only
        self.test_lr = test_lr
        self.test_wd = test_wd
        
        if self.head_only:  # freeze all but head
            if self.engine.model_str == "labram":
                self.engine.model = freeze_all_but_head_labram(self.engine.model)
            elif self.engine.model_str == "deepconvnet":
                self.engine.model = freeze_all_but_head_deepconvnet(self.engine.model)
            else:
                raise ValueError(f"head_only option not implemented for {self.engine.model_str}")
            
        
        self._allow_trainable = {
            n for n, p in self.engine.model.named_parameters() if p.requires_grad
        }
            
        
      

    def _forward_with(self, params_dict, x):
        if self.engine.use_amp:
            with torch.autocast(device_type=self.device.type):
                return functional_call(
                    self.engine.model,
                    params_dict,
                    args=(),
                    kwargs={"x": x, "electrodes": self.engine.electrodes},
                )
        else:
            return functional_call(
                self.engine.model,
                params_dict,
                args=(),
                kwargs={"x": x, "electrodes": self.engine.electrodes},
            )

    def _clone_as_leaf(self, params):
        return [
            torch.empty_like(p).copy_(p.detach()).requires_grad_(True) for p in params
        ]

    def _zero_shot_evaluate(
        self, subject_id: int, rep: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on subject without any adaptation."""
        self.engine.model.eval()

        # Get all available trials for this subject as query
        # Use the EpisodeIndex's subj_local_to_global mapping
        available_trials = list(
            range(len(self.test_epi.subj_local_to_global[subject_id]))
        )

        # print(f"Available trials: {len(available_trials)}")

        Xq, yq = fetch_by_indices(
            self.test_ds,
            self.test_epi,
            subject_id,
            available_trials,
            self.engine.device,
            self.engine.non_blocking,
        )

        with torch.no_grad():
            logits = self.engine._forward(Xq)
            loss = nn.functional.cross_entropy(logits, yq)
            accuracy = (logits.argmax(1) == yq).float().mean()

        self._log_to_console(
            {
                "subject_id": subject_id,
                "n_shots": 0,
                "n_repeat": rep,
                "n_query": len(yq),
                "epoch": 0,
                "loss": loss.item(),
                "accuracy": accuracy.item(),
            }
        )

        return {"accuracy": accuracy.item(), "loss": loss.item(), "n_samples": len(yq)}

    def _log_to_console(self, metrics: Dict[str, float]):
        line = [
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ]
        line.append(f"Runtime: {time.time() - self._t0:.2f}s")
        self._t0 = time.time()
        logger.info(" | ".join(line))

    def _get_support_and_query_trials(
        self, subject_id: int, n_shots: int
    ) -> Tuple[List[int], List[int]]:
        """
        Get n_shots trials for support and all remaining trials for query.

        Args:
            subject_id: Subject ID
            n_shots: Number of trials to use for support (per class)

        Returns:
            Tuple of (support_indices, query_indices)
        """
        # Get all available trials for this subject
        all_trials = self.test_epi.subj_local_to_global[subject_id]

        # Group trials by class
        trials_by_class = {}
        for trial_idx in all_trials:
            # Get the class for this trial by looking up in the dataset
            x, y, sid = self.test_ds[int(trial_idx)]
            class_id = y.item()

            if class_id not in trials_by_class:
                trials_by_class[class_id] = []
            trials_by_class[class_id].append(trial_idx)

        # Sample support trials (n_shots per class)
        support_indices = []
        query_indices = []

        for class_id, class_trials in trials_by_class.items():
            # Shuffle trials for this class
            self.rng.shuffle(class_trials)

            # Take n_shots for support, rest for query
            n_support = min(n_shots, len(class_trials))
            support_indices.extend(class_trials[:n_support])
            query_indices.extend(class_trials[n_support:])

        # Convert global indices to local indices for fetch_by_indices
        global_to_local = {
            global_idx: local_idx for local_idx, global_idx in enumerate(all_trials)
        }

        support_local = [global_to_local[idx] for idx in support_indices]
        query_local = [global_to_local[idx] for idx in query_indices]

        return support_local, query_local

    def _adapt_and_evaluate(
        self, subject_id: int, n_shots: int, n_epochs: int, rep: Optional[int] = None
    ) -> Dict[str, float]:
        """Adapt model on n_shots and evaluate on remaining trials."""
        self.engine.model.eval()

        # Get support and query trials
        sup_idx, que_idx = self._get_support_and_query_trials(subject_id, n_shots)

        # print(f"Support indices: {len(sup_idx)} trials")
        # print(f"Query indices: {len(que_idx)} trials")

        # Fetch data
        Xs, ys = fetch_by_indices(
            self.test_ds,
            self.test_epi,
            subject_id,
            sup_idx,
            self.engine.device,
            self.engine.non_blocking,
        )
        Xq, yq = fetch_by_indices(
            self.test_ds,
            self.test_epi,
            subject_id,
            que_idx,
            self.engine.device,
            self.engine.non_blocking,
        )

        # Get trainable parameters
        base_named = [
            (n, p)
            for n, p in self.engine.model.named_parameters()
            if n in self._allow_trainable
        ]
        base_names = [n for n, _ in base_named]
        base_params = [p for _, p in base_named]

        # Clone parameters for adaptation
        fast = self._clone_as_leaf(base_params)

        # Create optimizer for adaptation
        # print(self.test_lr, self.test_wd)
        adapter_optimizer = self.optimizer_factory(fast, lr=self.test_lr, wd=self.test_wd)
        

        loss_s_list = []
        loss_q_list = []
        accuracy_s_list = []
        accuracy_q_list = []
        runtime_list = []

        # Adaptation loop
        for epoch in range(n_epochs):
            epoch_t0 = time.time()
            adapter_optimizer.zero_grad()

            # Create parameter dict for functional call
            fast_dict = dict(zip(base_names, fast))

            # Forward pass on support set
            logits_s = self._forward_with(fast_dict, Xs)
            loss_s = nn.functional.cross_entropy(logits_s, ys)
            accuracy_s = (logits_s.argmax(1) == ys).float().mean()

            loss_s_list.append(loss_s.item())
            accuracy_s_list.append(accuracy_s.item())

            # Backward pass
            loss_s.backward()
            adapter_optimizer.step()
            
            runtime_list.append(time.time() - epoch_t0)
            # Evaluate on query set
            fast_dict = dict(zip(base_names, fast))
            with torch.no_grad():
                logits_q = self._forward_with(fast_dict, Xq)
                loss_q = nn.functional.cross_entropy(logits_q, yq)
                accuracy_q = (logits_q.argmax(1) == yq).float().mean()

            loss_q_list.append(loss_q.item())
            accuracy_q_list.append(accuracy_q.item())
            

            if self.use_wandb:
                self.engine.wandb_run.log(
                    {
                        f"{self.wandb_prefix}/s{subject_id}/{n_shots}shots/{epoch}epoch/loss_s": loss_s.item(),
                        f"{self.wandb_prefix}/s{subject_id}/{n_shots}shots/{epoch}epoch/accuracy_s": accuracy_s.item(),
                        f"{self.wandb_prefix}/s{subject_id}/{n_shots}shots/{epoch}epoch/loss_q": loss_q.item(),
                        f"{self.wandb_prefix}/s{subject_id}/{n_shots}shots/{epoch}epoch/accuracy_q": accuracy_q.item(),
                    }
                )

            self._log_to_console(
                {
                    "subject_id": subject_id,
                    "n_shots": n_shots,
                    "n_repeat": rep,
                    "epoch": epoch + 1,
                    "n_support": len(ys),
                    "n_query": len(yq),
                    "loss_s": loss_s.item(),
                    "accuracy_s": accuracy_s.item(),
                    "loss_q": loss_q.item(),
                    "accuracy_q": accuracy_q.item(),
                }
            )

        return {
            "accuracy_s": accuracy_s_list,
            "accuracy_q": accuracy_q_list,
            "loss_s": loss_s_list,
            "loss_q": loss_q_list,
            "n_support": len(ys),
            "n_query": len(yq),
            "epoch_runtimes": runtime_list,
        }
        
    def _adapt_with_cv_epoch_selection(
        self,
        subject_id: int,
        n_shots: int,
        max_epochs: int = 10,
        val_frac: float = 0.2,
        min_val_per_class: int = 1,
        n_folds: Optional[int] = None,
        rep: Optional[int] = None,
    ):
        import numpy as np

        assert n_shots >= 2, "CV epoch selection is only used for shots >= 2."

        sup_idx, que_idx = self._get_support_and_query_trials(subject_id, n_shots)
        Xs, ys = fetch_by_indices(self.test_ds, self.test_epi, subject_id, sup_idx,
                                self.engine.device, self.engine.non_blocking)
        Xq, yq = fetch_by_indices(self.test_ds, self.test_epi, subject_id, que_idx,
                                self.engine.device, self.engine.non_blocking)

        y_np = ys.detach().cpu().numpy().tolist()
        # y_np are labels on the SUPPORT set (n_shots per class, or less if dataset is smaller)
        classes = sorted(set(y_np))
        per_class = {c: [i for i, yy in enumerate(y_np) if yy == c] for c in classes}

        # Fail fast if any class has < 2 support samples — cannot split into 1 train / 1 val
        for c, idxs in per_class.items():
            if len(idxs) < 2:
                # fallback: skip CV and just use max_epochs on full support
                return {"best_epochs": max_epochs, "cv_mean_val_by_epoch": [float("nan")] * max_epochs,
                        "final": self._adapt_and_evaluate(subject_id, n_shots, max_epochs, rep=rep)}

        # Nominal per-class k, then cap to leave ≥1 train per class
        val_counts = {}
        for c, idxs in per_class.items():
            n_c = len(idxs)
            k = max(min_val_per_class, int(round(val_frac * n_c)))  # e.g., 20% ⇒ 0.4 of 2 ⇒ 0, then bumped to 1
            k = min(k, n_c - 1)                                     # ensure at least 1 train remains
            val_counts[c] = k

        # Ensure total val size ≥ 2
        if sum(val_counts.values()) < 2:
            # bump the largest class by 1 if possible
            big = max(classes, key=lambda cc: len(per_class[cc]))
            if val_counts[big] < len(per_class[big]) - 1:
                val_counts[big] += 1
            else:
                # if we truly can't reach 2 in total, skip CV (too tiny)
                return {"best_epochs": max_epochs, "cv_mean_val_by_epoch": [float("nan")] * max_epochs,
                        "final": self._adapt_and_evaluate(subject_id, n_shots, max_epochs, rep=rep)}


        if n_folds is None:
            # per class: how many disjoint k-sized chunks fit
            per_class_folds = []
            for c, idxs in per_class.items():
                k = val_counts[c]
                # guard: k could be 0 if a class is empty (shouldn't happen after checks), but be safe
                per_class_folds.append(len(idxs) // max(1, k))
            n_folds = max(2, min(per_class_folds))

        base_named = [(n, p) for n, p in self.engine.model.named_parameters()
                    if n in self._allow_trainable]
        base_names = [n for n, _ in base_named]
        base_params = [p for _, p in base_named]

        def _clone_fast():
            return [torch.empty_like(p).copy_(p.detach()).requires_grad_(True) for p in base_params]

        ##### FIND BEST EPOCH WITH CV #####
        val_curves = []
        for fold in range(n_folds):
            val_idx, tr_idx = [], []
            for c in classes:
                idx = per_class[c]
                k = val_counts[c]
                start = (fold * k) % len(idx) if len(idx) else 0
                sel = idx[start:start + k] if start + k <= len(idx) else (idx[start:] + idx[:(start + k) % len(idx)])
                sel = sel[:k]
                sset = set(sel)
                val_idx.extend(sel)
                tr_idx.extend([i for i in idx if i not in sset])

            Xtr, ytr = Xs[tr_idx], ys[tr_idx]
            Xva, yva = Xs[val_idx], ys[val_idx]

            fast = _clone_fast()
            opt = self.optimizer_factory(fast, lr=self.test_lr, wd=self.test_wd)

            fold_curve = []
            for _epoch in range(max_epochs):
                opt.zero_grad()
                fd = dict(zip(base_names, fast))
                logits_tr = self._forward_with(fd, Xtr)
                loss = nn.functional.cross_entropy(logits_tr, ytr)
                loss.backward()
                opt.step()

                fd = dict(zip(base_names, fast))
                with torch.no_grad():
                    logits_va = self._forward_with(fd, Xva)
                    acc_va = (logits_va.argmax(1) == yva).float().mean().item()
                fold_curve.append(acc_va)

            val_curves.append(fold_curve)

        mean_by_epoch = np.array(val_curves).mean(axis=0)
        std_by_epoch = np.array(val_curves).std(axis=0, ddof=1)
        best_epochs = int(np.argmax(mean_by_epoch)) + 1
        
        fold_best_epochs = [int(np.argmax(curve)) + 1 for curve in val_curves] # list of best epochs per fold
        within_range = (max(fold_best_epochs) - min(fold_best_epochs)) if fold_best_epochs else float("nan") # range of best epochs across folds
        within_std = (np.std(fold_best_epochs, ddof=1) if len(fold_best_epochs) > 1 else 0.0) # this is std of best epoch across folds
        
        print("VAL CURVES (folds):", val_curves) # dim: n_folds x max_epochs
        print("CV best epochs (folds):", fold_best_epochs) # dim: n_folds
        print("CV actual accuracy of these epochs (folds):", [val_curves[i][be-1] for i, be in enumerate(fold_best_epochs)]) # dim n_folds
        print("CV within range (folds):", within_range) # dim 1
        print("CV within std (folds):", within_std) # dim 1
        
        print("mean val acc by epoch:", mean_by_epoch) # dim: max_epochs
        print("std  val acc by epoch:", std_by_epoch)  # dim: max_epochs

        # print("val curves", val_curves)
        # print("mean by epoch", mean_by_epoch)
        print("best epochs", best_epochs, "/", max_epochs, "for shots", n_shots, "subject", subject_id, "rep", rep, "found with CV with", n_folds, "folds")

        ##### FINAL RUN WITH ALL SUPPORT for  #####
        final = self._adapt_and_evaluate(subject_id, n_shots, best_epochs, rep=rep)
        return {
                "best_epochs": best_epochs,
                "cv_mean_val_by_epoch": mean_by_epoch.tolist(),
                "cv_std_val_by_epoch": std_by_epoch.tolist(),
                "fold_best_epochs": fold_best_epochs,
                "within_cv_range": within_range,
                "within_cv_std": float(within_std),
                "n_folds": len(val_curves),
                "final": final,
            }

        
    def _evaluate_subject_once(
        self,
        subject_id: int,
        shots_list: List[int],
        n_epochs: int,
        rep: int,
        use_cv_epoch_selection: bool = False,
        cv_min_shots: int = 6,
        cv_val_frac: float = 0.2,
        cv_min_per_class: int = 1,
        cv_n_folds: Optional[int] = None,
    ) -> List[Dict]:
        def _pad(seq, N, pad_val=float("nan")):
            seq = list(seq)
            return seq[:N] if len(seq) >= N else seq + [pad_val] * (N - len(seq))
        def _pad_time(seq, N):
            return _pad(seq, N, pad_val=0.0)

        rows = []

        for n_shots in shots_list:
            if n_shots == 0:
                result = self._zero_shot_evaluate(subject_id, rep=rep)
                acc_test = [result["accuracy"]] * n_epochs
                loss_q = [result["loss"]] * n_epochs
                acc_train = [float("nan")] * n_epochs
                loss_supp = [float("nan")] * n_epochs

                row = {
                        "subject_id": subject_id,
                        "shots": 0,
                        "repetition": rep,
                        "final_accuracy": result["accuracy"],
                        "final_loss": result["loss"],
                        "final_accuracy_train": float("nan"),
                        "n_samples": result["n_samples"],
                        "n_support": 0,
                        "n_query": result["n_samples"],
                        "cv_used": False,
                        "cv_chosen_best_epochs": float("nan"),
                        "cv_n_folds": float("nan"),
                        "cv_fold_best_epochs": json.dumps([]),
                        "cv_within_cv_range": float("nan"),
                        "cv_within_cv_std": float("nan"),
                        "cv_mean_val_by_epoch": json.dumps([]),
                        "cv_std_val_by_epoch": json.dumps([]),
                    }
                for ei in range(n_epochs):
                    row[f"acc_train_e{ei+1}"] = acc_train[ei]
                    row[f"acc_test_e{ei+1}"]  = acc_test[ei]
                    row[f"loss_supp_e{ei+1}"] = loss_supp[ei]
                    row[f"loss_q_e{ei+1}"]    = loss_q[ei]
                    row[f"epoch_time_e{ei+1}"] = 0.0
                rows.append(row)
                continue

            # few-shot branch
            available_trials = len(self.test_epi.subj_local_to_global[subject_id])
            assert n_shots <= available_trials, (
                f"Subject {subject_id}: Requested {n_shots} shots but only {available_trials} trials available."
            )

            # optional CV epoch selection
            if use_cv_epoch_selection and n_shots >= cv_min_shots:
                cv_out = self._adapt_with_cv_epoch_selection(
                    subject_id, n_shots, max_epochs=n_epochs,
                    val_frac=cv_val_frac, min_val_per_class=cv_min_per_class,
                    n_folds=cv_n_folds, rep=rep
                )
                run = cv_out["final"]
                chosen_epochs = cv_out["best_epochs"]

                cv_cols = {
                    "cv_used": True,
                    "cv_chosen_best_epochs": int(chosen_epochs),
                    "cv_n_folds": int(cv_out.get("n_folds")) if cv_out.get("n_folds") is not None else float("nan"),
                    "cv_fold_best_epochs": json.dumps(cv_out.get("fold_best_epochs")),
                    "cv_within_cv_range": float(cv_out.get("within_cv_range")),
                    "cv_within_cv_std": float(cv_out.get("within_cv_std")),
                    "cv_mean_val_by_epoch": json.dumps(cv_out.get("cv_mean_val_by_epoch")),
                    "cv_std_val_by_epoch": json.dumps(cv_out.get("cv_std_val_by_epoch")),
                }
            else:
                run = self._adapt_and_evaluate(subject_id, n_shots, n_epochs, rep=rep)
                chosen_epochs = n_epochs
                cv_cols = {
                    "cv_used": False,
                    "cv_chosen_best_epochs": float("nan"),
                    "cv_n_folds": float("nan"),
                    "cv_fold_best_epochs": json.dumps([]),
                    "cv_within_cv_range": float("nan"),
                    "cv_within_cv_std": float("nan"),
                    "cv_mean_val_by_epoch": json.dumps([]),
                    "cv_std_val_by_epoch": json.dumps([]),
                }


            acc_q = run.get("accuracy_q")
            acc_s = run.get("accuracy_s")
            loss_q = run.get("loss_q")
            loss_s = run.get("loss_s")
            epoch_runtimes = run.get("epoch_runtimes")

            row = {
                    "subject_id": subject_id,
                    "shots": n_shots,
                    "repetition": rep,
                    "final_accuracy": (acc_q[-1] if acc_q else float("nan")),
                    "final_loss": (loss_q[-1] if loss_q else float("nan")),
                    "final_accuracy_train": (acc_s[-1] if acc_s else float("nan")),
                    "n_samples": run.get("n_support") + run.get("n_query"),
                    "n_support": run.get("n_support"),
                    "n_query": run.get("n_query"),
                    **cv_cols,
                }
            # pad back to n_epochs so the saved schema stays identical
            acc_q_pad  = _pad(acc_q,  n_epochs)
            acc_s_pad  = _pad(acc_s,  n_epochs)
            loss_q_pad = _pad(loss_q, n_epochs)
            loss_s_pad = _pad(loss_s, n_epochs)
            time_pad   = _pad_time(epoch_runtimes, n_epochs)

            for ei in range(n_epochs):
                row[f"acc_train_e{ei+1}"] = acc_s_pad[ei]
                row[f"acc_test_e{ei+1}"]  = acc_q_pad[ei]
                row[f"loss_supp_e{ei+1}"] = loss_s_pad[ei]
                row[f"loss_q_e{ei+1}"]    = loss_q_pad[ei]
                row[f"epoch_time_e{ei+1}"] = time_pad[ei]

            rows.append(row)

        return rows


    def test_all_subjects(
        self,
        shots_list: List[int] = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25],
        n_epochs: int = 10,
        n_repeats: int = 10,
        use_cv_epoch_selection: bool = False,
        cv_min_shots: int = 6,
        cv_val_frac: float = 0.2,
        cv_min_per_class: int = 1,
        cv_n_folds: Optional[int] = None, # will be automatically determined if None
    ) -> Dict[int, pd.DataFrame]:
        logger.info(f"Testing {len(self.S_test)} subjects with {n_epochs} epochs")
        all_results = {}
        aggregated_results = []
        repetition_rows = []
        all_cv_rows = []

        for subject_id in self.S_test:
            start_time = time.time()
            subject_results_for_return = {}

            # run all repetitions through the SAME core path as test_subject_adaptation
            per_shot_records_accumulator = {shots: [] for shots in shots_list}

            for rep in range(n_repeats):
                rows = self._evaluate_subject_once(
                    subject_id=subject_id,
                    shots_list=shots_list,
                    n_epochs=n_epochs,
                    rep=rep,
                    use_cv_epoch_selection=use_cv_epoch_selection,
                    cv_min_shots=cv_min_shots,
                    cv_val_frac=cv_val_frac,
                    cv_min_per_class=cv_min_per_class,
                    cv_n_folds=cv_n_folds,
                )
                repetition_rows.extend(rows)
                # stash per-shot rows for this subject
                for r in rows:
                    per_shot_records_accumulator[r["shots"]].append(r)

            # build return dict (per-shot DataFrame indexed by repetition)
            for shots in shots_list:
                shot_rows = per_shot_records_accumulator.get(shots)
                if not shot_rows:
                    continue
                shot_df = pd.DataFrame(shot_rows).set_index("repetition")
                subject_results_for_return[shots] = shot_df

            all_results[subject_id] = subject_results_for_return

            # Save cross-validation results (JSON of the CV rows)
            if self.save_dir and all_cv_rows:
                cv_serializable = {
                    "subject_id": subject_id,
                    "cv_results": [r.to_dict("index") for r in all_cv_rows],
                }
                with open(
                    self.save_dir / f"subject_{subject_id}_cv_results.json", "w"
                ) as f:
                    json.dump(cv_serializable, f, indent=2)

            # Save individual subject results (JSON of the per-shot dict of repetition DataFrames)
            if self.save_dir:
                serializable = {
                    "subject_id": subject_id,
                    "results": {
                        k: v.to_dict("index")
                        for k, v in subject_results_for_return.items()
                    },
                    "shots_list": shots_list,
                    "n_epochs": n_epochs,
                    "n_repeats": n_repeats,
                    "timestamp": time.time(),
                }
                with open(
                    self.save_dir / f"subject_{subject_id}_results.json", "w"
                ) as f:
                    json.dump(serializable, f, indent=2)

            # Log to wandb if requested: use mean final_accuracy across repetitions for each shot
            if self.use_wandb and self.engine.wandb_run is not None:
                for shots in shots_list:
                    shot_df = subject_results_for_return.get(shots)
                    if shot_df is not None and not shot_df.empty:
                        mean_final_acc = shot_df["final_accuracy"].mean()
                        if not pd.isna(mean_final_acc):
                            self.engine.wandb_run.log(
                                {
                                    f"{self.wandb_prefix}/subject_{subject_id}/shots_{shots}/final_accuracy": float(
                                        mean_final_acc
                                    )
                                }
                            )

            # Collect for aggregation: use per-repetition final accuracies
            for shots in shots_list:
                shot_df = all_results[subject_id].get(shots)
                if shot_df is not None and not shot_df.empty:
                    for rep_idx, row in shot_df.iterrows():
                        final_acc = row.get("final_accuracy")
                        if not pd.isna(final_acc):
                            aggregated_results.append(
                                {
                                    "subject_id": subject_id,
                                    "shots": shots,
                                    "epochs": n_epochs,
                                    "final_accuracy": float(final_acc),
                                }
                            )

            elapsed = time.time() - start_time
            logger.info(f"Subject {subject_id} completed in {elapsed:.2f}s")

        # Log aggregated results to wandb (simplified - only summary stats)
        if self.use_wandb and self.engine.wandb_run is not None:
            agg_df = pd.DataFrame(aggregated_results)

            # Only log final summary statistics, not per-subject details
            for shots in shots_list:
                subset = agg_df[agg_df["shots"] == shots]
                if not subset.empty:
                    mean_acc = subset["final_accuracy"].mean()
                    std_acc = subset["final_accuracy"].std()
                    n_subjects = len(subset)

                    # Log only the essential summary metrics
                    self.engine.wandb_run.log(
                        {
                            f"{self.wandb_prefix}/summary/shots_{shots}/mean_accuracy": mean_acc,
                            f"{self.wandb_prefix}/summary/shots_{shots}/std_accuracy": std_acc,
                            f"{self.wandb_prefix}/summary/shots_{shots}/n_subjects": n_subjects,
                        }
                    )

            # Log a single summary table for easy comparison
            summary_table = []
            for shots in shots_list:
                subset = agg_df[agg_df["shots"] == shots]
                if not subset.empty:
                    summary_table.append(
                        {
                            "shots": shots,
                            "mean_accuracy": f"{subset['final_accuracy'].mean():.3f}",
                            "std_accuracy": f"{subset['final_accuracy'].std():.3f}",
                            "n_subjects": len(subset),
                        }
                    )

            if summary_table:
                self.engine.wandb_run.log(
                    {
                        f"{self.wandb_prefix}/summary_table": wandb.Table(
                            data=[
                                [
                                    row["shots"],
                                    row["mean_accuracy"],
                                    row["std_accuracy"],
                                    row["n_subjects"],
                                ]
                                for row in summary_table
                            ],
                            columns=[
                                "shots",
                                "mean_accuracy",
                                "std_accuracy",
                                "n_subjects",
                            ],
                        )
                    }
                )
        # Save aggregated results
        if self.save_dir:
            summary = {
                "experiment_name": self.engine.experiment_name,
                "test_subjects": self.S_test,
                "shots_list": shots_list,
                "n_epochs": n_epochs,
                "aggregated_results": aggregated_results,
                "timestamp": time.time(),
            }

            with open(self.save_dir / "aggregated_results.json", "w") as f:
                json.dump(summary, f, indent=2)
        # Save repetition-level CSV (rows: subject x shots x repetition; columns include epoch metrics)
        if repetition_rows and self.save_dir:
            rep_df = pd.DataFrame(repetition_rows)

            # Reorder columns so that per-epoch metrics are grouped as:
            # acc_train_e1..eN, acc_test_e1..eN, loss_supp_e1..eN, loss_q_e1..eN
            try:
                n_epochs = int(n_epochs)
            except Exception:
                # fallback: infer epoch columns count from column names
                epoch_idxs = sorted(
                    set(
                        int(c.split("e")[-1])
                        for c in rep_df.columns
                        if (
                            c.startswith("acc_train_e")
                            or c.startswith("acc_test_e")
                            or c.startswith("loss_supp_e")
                            or c.startswith("loss_q_e")
                        )
                    )
                )
                n_epochs = max(epoch_idxs) if epoch_idxs else 0

            base_cols = [
                "subject_id",
                "shots",
                "repetition",
                "final_accuracy",
                "final_loss",
                "final_accuracy_train",
                "n_samples",
                "n_support",
                "n_query",
            ]

            acc_train_cols = [f"acc_train_e{i+1}" for i in range(n_epochs)]
            acc_test_cols = [f"acc_test_e{i+1}" for i in range(n_epochs)]
            loss_train_cols = [f"loss_supp_e{i+1}" for i in range(n_epochs)]
            loss_test_cols = [f"loss_q_e{i+1}" for i in range(n_epochs)]

            desired = (
                base_cols
                + acc_train_cols
                + acc_test_cols
                + loss_train_cols
                + loss_test_cols
            )

            # Keep only columns that exist in the DataFrame (be robust to missing columns)
            ordered = [c for c in desired if c in rep_df.columns]
            # Add any other columns that weren't anticipated at the end
            remaining = [c for c in rep_df.columns if c not in ordered]
            rep_df = rep_df[ordered + remaining]

            rep_csv = self.save_dir / "repetition_results.csv"
            rep_df.to_csv(rep_csv, index=False)
            logger.info(f"Saved repetition-level CSV to {rep_csv}")

        self.plot_aggregated_results(
            self.save_dir / "aggregated_results.json",
            metric="final_accuracy",
            save_path=self.save_dir / "plot_final_accuracy.png",
        )

        logger.info(
            f"Saved aggregated results to {self.save_dir / 'aggregated_results.json'}"
        )
        logger.info(f"Saved plot to {self.save_dir / 'plot_final_accuracy.png'}")

        logger.info(f"Testing completed for all {len(self.S_test)} subjects")
        return all_results

    def plot_aggregated_results(
        self,
        json_path: Union[str, Path],
        metric: str = "final_accuracy",
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ):
        """
        Plot per-subject lines and the average line across shots from aggregated_results.json.

        Args:
            json_path: Path to aggregated_results.json (produced by TestEngine.test_all_subjects).
            metric: Metric key inside aggregated results to plot (default: "final_accuracy").
            title: Optional plot title. Defaults to experiment_name if present.
            save_path: Where to save the plot PNG. Defaults next to json: "plot_{metric}.png".
            show: Whether to call plt.show().

        Returns:
            (fig, ax): Matplotlib figure and axis.
        """
        import json
        from pathlib import Path
        import matplotlib.pyplot as plt
        import pandas as pd

        json_path = Path(json_path)
        with open(json_path, "r") as f:
            summary = json.load(f)

        exp_name = summary.get("experiment_name")
        shots_list = summary.get("shots_list")
        agg = pd.DataFrame(summary.get("aggregated_results"))

        if agg.empty:
            raise ValueError("No aggregated_results found in the JSON.")

        # Ensure expected columns exist
        required_cols = {"subject_id", "shots", metric}
        if not required_cols.issubset(agg.columns):
            missing = required_cols - set(agg.columns)
            raise ValueError(
                f"Missing required columns in aggregated_results: {missing}"
            )

        # Pivot for easier plotting; one subject per 'line'
        # Not all subjects necessarily have all shots; leave missing as NaN
        pivot = agg.pivot_table(
            index="shots", columns="subject_id", values=metric, aggfunc="mean"
        ).sort_index()

        # Compute mean across subjects for each shot
        mean_series = pivot.mean(axis=1)

        # Prepare plot
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        # Plot each subject line
        # We avoid adding a legend entry per subject to keep it clean
        ax.plot([], [], alpha=0.3, label="Per-subject")  # legend stub
        for sid in pivot.columns:
            ax.plot(pivot.index, pivot[sid], alpha=0.3, linewidth=1.5)

        # Plot average line on top
        ax.plot(
            mean_series.index,
            mean_series.values,
            color="black",
            linewidth=2.5,
            marker="o",
            label="Average",
            zorder=5,
        )

        # Cosmetic settings
        ax.set_xlabel("Shots")
        ax.set_ylabel(metric.replace("_", " ").title())
        if title is None:
            title = f"{exp_name} — {metric.replace('_', ' ').title()}"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        # Save
        if save_path is None:
            save_path = json_path.parent / f"plot_{metric}.png"
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=200)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

