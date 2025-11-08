import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import random
import torch
import torch.nn as nn
from torch.func import functional_call
import pandas as pd
from engines.utils import fetch_by_indices, build_episode_index

logger = logging.getLogger(__name__)


class TestEngine:
    def __init__(
        self,
        engine,
        test_ds,
        run_size: int = 100,
        save_dir: Union[str, Path] = None,
        test_lr: float = None,
        test_wd: float = None,
    ):
        self.engine = engine
        self.optimizer_factory = engine.optimizer_factory

        self.rng = random.Random(int(111))
        self.S_test = test_ds.subject_ids
        self.test_ds = test_ds
        self.test_epi = build_episode_index(self.test_ds, run_size=run_size)

        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        
        self.test_lr = test_lr
        self.test_wd = test_wd
            
        self._allow_trainable = {
            n for n, p in self.engine.model.named_parameters() if p.requires_grad
        }

    def _log_to_console(self, metrics: Dict[str, float]):
        line = [
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ]
        line.append(f"Runtime: {time.time() - self._t0:.2f}s")
        self._t0 = time.time()
        logger.info(" | ".join(line))
            
    def _forward_with(self, params_dict, x):
        if self.engine.use_amp:
            with torch.autocast(device_type=self.engine.device.type):
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
        

        
    def _evaluate_subject_once(
        self,
        subject_id: int,
        shots_list: List[int],
        n_epochs: int,
        rep: int,
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

            
            run = self._adapt_and_evaluate(subject_id, n_shots, n_epochs, rep=rep)

            acc_q = run.get("accuracy_q")
            acc_s = run.get("accuracy_s")
            loss_q = run.get("loss_q")
            loss_s = run.get("loss_s")
            epoch_runtimes = run.get("epoch_runtimes")

            row = {
                    "subject_id": subject_id,
                    "shots": n_shots*2, # total shots across classes
                    "repetition": rep,
                    "final_accuracy": (acc_q[-1] if acc_q else float("nan")),
                    "final_loss": (loss_q[-1] if loss_q else float("nan")),
                    "final_accuracy_train": (acc_s[-1] if acc_s else float("nan")),
                    "n_samples": run.get("n_support") + run.get("n_query"),
                    "n_support": run.get("n_support"),
                    "n_query": run.get("n_query"),
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
    ) -> Dict[int, pd.DataFrame]:
        logger.info(f"Testing {len(self.S_test)} subjects with {n_epochs} epochs")
        repetition_rows = []

        for subject_id in self.S_test:
            start_time = time.time()
            for rep in range(n_repeats):
                rows = self._evaluate_subject_once(
                    subject_id=subject_id,
                    shots_list=shots_list,
                    n_epochs=n_epochs,
                    rep=rep,
                )
                repetition_rows.extend(rows)
            elapsed = time.time() - start_time
            logger.info(f"Subject {subject_id} completed in {elapsed:.2f}s")

        # Save repetition-level CSV (rows: subject x shots x repetition; columns include epoch metrics)
        if repetition_rows and self.save_dir:
            rep_df = pd.DataFrame(repetition_rows)

            n_epochs = int(n_epochs)

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

        logger.info(f"Testing completed for all {len(self.S_test)} subjects")
