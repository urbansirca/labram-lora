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
from meta_helpers import (
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
                    "epoch": epoch,
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

    def test_subject_adaptation(
        self,
        subject_id: int,
        shots_list: List[int] = [0, 1, 2, 3, 4, 5, 10, 25, 50],
        n_epochs: int = 10,
    ) -> pd.DataFrame:
        """
        Test adaptation for a single subject across different shots.

        Args:
            subject_id: Subject ID to test
            shots_list: List of shot counts to test
            n_epochs: Number of adaptation epochs (single value, not a list)

        Returns:
            DataFrame with shots as rows, columns for different metrics
        """
        logger.info(f"Testing subject {subject_id} with {n_epochs} epochs")

        results = []

        for n_shots in shots_list:
            if n_shots == 0:
                # Zero-shot evaluation
                result = self._zero_shot_evaluate(subject_id)
                # For zero-shot, we only have final accuracy and loss
                results.append(
                    {
                        "shots": n_shots,
                        "epochs": n_epochs,
                        "final_accuracy": result["accuracy"],
                        "final_loss": result["loss"],
                        "n_samples": result["n_samples"],
                        "n_support": 0,  # No support set for zero-shot
                        "n_query": result["n_samples"],  # All samples are query
                        "accuracy_evolution": [result["accuracy"]] * n_epochs,
                        "loss_evolution": [result["loss"]] * n_epochs,
                        "support_accuracy_evolution": [float("nan")]
                        * n_epochs,  # No support set
                        "support_loss_evolution": [float("nan")]
                        * n_epochs,  # No support set
                        "epoch_time_evolution": [0.0] * n_epochs,  # No adaptation time
                    }
                )
            else:
                # Check if we have enough trials for this many shots
                available_trials = len(self.test_epi.subj_local_to_global[subject_id])
                assert (
                    n_shots <= available_trials
                ), f"Subject {subject_id}: Requested {n_shots} shots but only {available_trials} trials available."

                # Few-shot adaptation
                result = self._adapt_and_evaluate(subject_id, n_shots, n_epochs)
                results.append(
                    {
                        "shots": n_shots,
                        "epochs": n_epochs,
                        "final_accuracy": result["accuracy_q"][
                            -1
                        ],  # Last epoch accuracy
                        "final_loss": result["loss_q"][-1],  # Last epoch loss
                        "n_samples": result["n_support"]
                        + result["n_query"],  # Total samples
                        "n_support": result["n_support"],
                        "n_query": result["n_query"],
                        "accuracy_evolution": result["accuracy_q"],
                        "loss_evolution": result["loss_q"],
                        "support_accuracy_evolution": result["accuracy_s"],
                        "support_loss_evolution": result["loss_s"],
                        "epoch_time_evolution": result["epoch_runtimes"],
                    }
                )

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Set shots as index for easier access
        df = df.set_index("shots")

        return df

    def test_all_subjects(
        self,
        shots_list: List[int] = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100],
        n_epochs: int = 10,
        n_repeats: int = 10,
    ) -> Dict[int, pd.DataFrame]:
        """
        Test adaptation for all test subjects.

        Args:
            shots_list: List of shot counts to test
            n_epochs: Number of adaptation epochs (single value)
            save_dir: Directory to save results

        Returns:
            Dictionary mapping subject_id -> DataFrame of results
        """
                
        
        logger.info(f"Testing {len(self.S_test)} subjects with {n_epochs} epochs")
        all_results = {}
        aggregated_results = []

        # For repetition-level CSV: collect rows with columns for epoch-wise metrics
        repetition_rows = []

        for subject_id in self.S_test:
            start_time = time.time()

            # For each subject, iterate shots and repetitions
            subject_results_for_return = {}
            for n_shots in shots_list:
                per_shot_records = []

                if n_shots == 0:
                    # Zero-shot: repeat evaluation n_repeats times (identical results)
                    for rep in range(n_repeats):
                        result = self._zero_shot_evaluate(subject_id, rep=rep)
                        # Build row for CSV with epoch columns filled with same value
                        acc_test = [result["accuracy"]] * n_epochs
                        loss_q = [result["loss"]] * n_epochs
                        acc_train = [float("nan")] * n_epochs
                        loss_supp = [float("nan")] * n_epochs

                        row = {
                            "subject_id": subject_id,
                            "shots": n_shots,
                            "repetition": rep,
                            # keep existing final_accuracy as test final for compatibility
                            "final_accuracy": result["accuracy"],
                            "final_loss": result["loss"],
                            # also expose train final (NaN for zero-shot)
                            "final_accuracy_train": float("nan"),
                            "n_samples": result["n_samples"],
                            "n_support": 0,
                            "n_query": result["n_samples"],
                        }
                        # add epoch columns: train (support), test (query), support/test losses
                        for ei in range(n_epochs):
                            row[f"acc_train_e{ei+1}"] = acc_train[ei]
                            row[f"acc_test_e{ei+1}"] = acc_test[ei]
                            row[f"loss_supp_e{ei+1}"] = loss_supp[ei]
                            row[f"loss_q_e{ei+1}"] = loss_q[ei]
                            row[f"epoch_time_e{ei+1}"] = 0.0  # No adaptation time
                        repetition_rows.append(row)
                        per_shot_records.append(row)
                else:
                    # Few-shot: perform n_repeats different samplings/adaptations
                    available_trials = len(
                        self.test_epi.subj_local_to_global[subject_id]
                    )
                    assert (
                        n_shots <= available_trials
                    ), f"Subject {subject_id}: Requested {n_shots} shots but only {available_trials} trials available."

                    for rep in range(n_repeats):
                        run = self._adapt_and_evaluate(
                            subject_id, n_shots, n_epochs, rep=rep
                        )
                        acc_q = run["accuracy_q"]
                        acc_s = run.get("accuracy_s", [])
                        loss_q = run.get("loss_q", [])
                        loss_s = run.get("loss_s", [])
                        epoch_runtimes = run.get("epoch_runtimes", [])

                        row = {
                            "subject_id": subject_id,
                            "shots": n_shots,
                            "repetition": rep,
                            # keep existing final_accuracy as test final for compatibility
                            "final_accuracy": acc_q[-1] if len(acc_q) else float("nan"),
                            "final_loss": loss_q[-1] if len(loss_q) else float("nan"),
                            # final train/support accuracy
                            "final_accuracy_train": (
                                acc_s[-1] if len(acc_s) else float("nan")
                            ),
                            "n_samples": run.get("n_support", 0)
                            + run.get("n_query", 0),
                            "n_support": run.get("n_support", 0),
                            "n_query": run.get("n_query", 0),
                        }
                        # add epoch columns (pad/truncate to n_epochs)
                        for ei in range(n_epochs):
                            row[f"acc_train_e{ei+1}"] = (
                                acc_s[ei] if ei < len(acc_s) else float("nan")
                            )
                            row[f"acc_test_e{ei+1}"] = (
                                acc_q[ei] if ei < len(acc_q) else float("nan")
                            )
                            row[f"loss_supp_e{ei+1}"] = (
                                loss_s[ei] if ei < len(loss_s) else float("nan")
                            )
                            row[f"loss_q_e{ei+1}"] = (
                                loss_q[ei] if ei < len(loss_q) else float("nan")
                            )
                            row[f"epoch_time_e{ei+1}"] = (
                                epoch_runtimes[ei] if ei < len(epoch_runtimes) else float("nan")
                            )
                        repetition_rows.append(row)
                        per_shot_records.append(row)

                # Aggregate final accuracy per-shot across repetitions for summary
                # Build a small DataFrame for this subject/shot where index is repetition
                shot_df = pd.DataFrame(per_shot_records).set_index("repetition")
                subject_results_for_return[n_shots] = shot_df

            all_results[subject_id] = subject_results_for_return

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

        exp_name = summary.get("experiment_name", "")
        shots_list = summary.get("shots_list", [])
        agg = pd.DataFrame(summary.get("aggregated_results", []))

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
