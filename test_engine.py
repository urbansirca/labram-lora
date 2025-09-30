import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Tuple
import wandb

import torch
import torch.nn as nn
import pandas as pd
from meta_helpers import (
    sample_support,
    sample_query,
    fetch_by_indices,
)

logger = logging.getLogger(__name__)


class TestEngine:
    def __init__(
        self,
        meta_engine,
        optimizer_factory: Optional[Callable[[List], torch.optim.Optimizer]] = None,
        use_wandb: bool = False,
        wandb_prefix: str = "test",
        experiment_name: str = "test",
    ):
        """
        Test class for evaluating model adaptation on test subjects.

        Args:
            meta_engine: MetaEngine instance with trained model
            optimizer_factory: Factory for creating optimizers for adaptation.
                             If None, uses meta_engine.optimizer_factory
            use_wandb: Whether to log results to wandb
            wandb_prefix: Prefix for wandb logging keys
        """
        self.meta_engine = meta_engine
        self.optimizer_factory = optimizer_factory or meta_engine.optimizer_factory
        self.use_wandb = use_wandb
        self.wandb_prefix = wandb_prefix
        self.rng = meta_engine.rng
        self.experiment_name = experiment_name

        self.save_dir = Path("results/test/{}".format(self.experiment_name))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def _zero_shot_evaluate(self, subject_id: int) -> Dict[str, float]:
        """Evaluate model on subject without any adaptation."""
        self.meta_engine.model.eval()

        # Get all available trials for this subject as query
        # Use the EpisodeIndex's subj_local_to_global mapping
        available_trials = list(
            range(len(self.meta_engine.test_epi.subj_local_to_global[subject_id]))
        )

        # print(f"Available trials: {len(available_trials)}")

        Xq, yq = fetch_by_indices(
            self.meta_engine.test_ds,
            self.meta_engine.test_epi,
            subject_id,
            available_trials,
            self.meta_engine.device,
            self.meta_engine.non_blocking,
        )

        with torch.no_grad():
            logits = self.meta_engine._forward(Xq)
            loss = nn.functional.cross_entropy(logits, yq)
            accuracy = (logits.argmax(1) == yq).float().mean()

        self._log_to_console(
            {
                "subject_id": subject_id,
                "n_shots": 0,
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
        all_trials = self.meta_engine.test_epi.subj_local_to_global[subject_id]

        # Group trials by class
        trials_by_class = {}
        for trial_idx in all_trials:
            # Get the class for this trial by looking up in the dataset
            x, y, sid = self.meta_engine.test_ds[int(trial_idx)]
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
        self, subject_id: int, n_shots: int, n_epochs: int
    ) -> Dict[str, float]:
        """Adapt model on n_shots and evaluate on remaining trials."""
        self.meta_engine.model.eval()

        # Get support and query trials
        sup_idx, que_idx = self._get_support_and_query_trials(subject_id, n_shots)

        # print(f"Support indices: {len(sup_idx)} trials")
        # print(f"Query indices: {len(que_idx)} trials")

        # Fetch data
        Xs, ys = fetch_by_indices(
            self.meta_engine.test_ds,
            self.meta_engine.test_epi,
            subject_id,
            sup_idx,
            self.meta_engine.device,
            self.meta_engine.non_blocking,
        )
        Xq, yq = fetch_by_indices(
            self.meta_engine.test_ds,
            self.meta_engine.test_epi,
            subject_id,
            que_idx,
            self.meta_engine.device,
            self.meta_engine.non_blocking,
        )

        # Get trainable parameters
        base_named = [
            (n, p)
            for n, p in self.meta_engine.model.named_parameters()
            if n in self.meta_engine._allow_trainable
        ]
        base_names = [n for n, _ in base_named]
        base_params = [p for _, p in base_named]

        # Clone parameters for adaptation
        fast = self.meta_engine._clone_as_leaf(base_params)

        # Create optimizer for adaptation
        adapter_optimizer = self.optimizer_factory(fast)

        loss_s_list = []
        loss_q_list = []
        accuracy_s_list = []
        accuracy_q_list = []

        # Adaptation loop
        for epoch in range(n_epochs):
            adapter_optimizer.zero_grad()

            # Create parameter dict for functional call
            fast_dict = dict(zip(base_names, fast))

            # Forward pass on support set
            logits_s = self.meta_engine._forward_with(fast_dict, Xs)
            loss_s = nn.functional.cross_entropy(logits_s, ys)
            accuracy_s = (logits_s.argmax(1) == ys).float().mean()

            loss_s_list.append(loss_s.item())
            accuracy_s_list.append(accuracy_s.item())

            # Backward pass
            loss_s.backward()
            adapter_optimizer.step()

            # Evaluate on query set
            fast_dict = dict(zip(base_names, fast))
            with torch.no_grad():
                logits_q = self.meta_engine._forward_with(fast_dict, Xq)
                loss_q = nn.functional.cross_entropy(logits_q, yq)
                accuracy_q = (logits_q.argmax(1) == yq).float().mean()

            loss_q_list.append(loss_q.item())
            accuracy_q_list.append(accuracy_q.item())

            if self.use_wandb:
                self.meta_engine.wandb_run.log(
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
                    }
                )
            else:
                # Check if we have enough trials for this many shots
                available_trials = len(
                    self.meta_engine.test_epi.subj_local_to_global[subject_id]
                )
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
        experiment_name: str = "test",
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
        logger.info(
            f"Testing {len(self.meta_engine.S_test)} subjects with {n_epochs} epochs"
        )

        all_results = {}
        aggregated_results = []

        for subject_id in self.meta_engine.S_test:
            start_time = time.time()

            # Test this subject
            subject_df = self.test_subject_adaptation(subject_id, shots_list, n_epochs)
            all_results[subject_id] = subject_df

            # Save individual subject results
            if self.save_dir:
                subject_results = {
                    "subject_id": subject_id,
                    "results": subject_df.to_dict(
                        "index"
                    ),  # Convert to dict for JSON serialization
                    "shots_list": shots_list,
                    "n_epochs": n_epochs,
                    "timestamp": time.time(),
                }

                with open(
                    self.save_dir / f"subject_{subject_id}_results.json", "w"
                ) as f:
                    json.dump(subject_results, f, indent=2)

            # Log to wandb if requested
            if self.use_wandb and self.meta_engine.wandb_run is not None:
                # Log individual subject metrics
                for shots in shots_list:
                    if shots in subject_df.index:
                        final_acc = subject_df.loc[shots, "final_accuracy"]
                        if not pd.isna(final_acc):
                            self.meta_engine.wandb_run.log(
                                {
                                    f"{self.wandb_prefix}/subject_{subject_id}/shots_{shots}/final_accuracy": final_acc
                                }
                            )

            # Collect for aggregation
            for shots in shots_list:
                if shots in subject_df.index:
                    final_acc = subject_df.loc[shots, "final_accuracy"]
                    if not pd.isna(final_acc):
                        aggregated_results.append(
                            {
                                "subject_id": subject_id,
                                "shots": shots,
                                "epochs": n_epochs,
                                "final_accuracy": final_acc,
                            }
                        )

            elapsed = time.time() - start_time
            logger.info(f"Subject {subject_id} completed in {elapsed:.2f}s")

        # Log aggregated results to wandb (simplified - only summary stats)
        if self.use_wandb and self.meta_engine.wandb_run is not None:
            agg_df = pd.DataFrame(aggregated_results)

            # Only log final summary statistics, not per-subject details
            for shots in shots_list:
                subset = agg_df[agg_df["shots"] == shots]
                if not subset.empty:
                    mean_acc = subset["final_accuracy"].mean()
                    std_acc = subset["final_accuracy"].std()
                    n_subjects = len(subset)

                    # Log only the essential summary metrics
                    self.meta_engine.wandb_run.log(
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
                self.meta_engine.wandb_run.log(
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
                "experiment_name": self.meta_engine.experiment_name,
                "test_subjects": self.meta_engine.S_test,
                "shots_list": shots_list,
                "n_epochs": n_epochs,
                "aggregated_results": aggregated_results,
                "timestamp": time.time(),
            }

            with open(self.save_dir / "aggregated_results.json", "w") as f:
                json.dump(summary, f, indent=2)
        
        self.plot_aggregated_results(
            self.save_dir / "aggregated_results.json",
            metric="final_accuracy",
            save_path=self.save_dir / "plot_final_accuracy.png",
        )

        logger.info(f"Saved aggregated results to {self.save_dir / 'aggregated_results.json'}")
        logger.info(f"Saved plot to {self.save_dir / 'plot_final_accuracy.png'}")

        logger.info(
            f"Testing completed for all {len(self.meta_engine.S_test)} subjects"
        )
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
            raise ValueError(f"Missing required columns in aggregated_results: {missing}")

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


# # Example usage function
# def run_adaptation_test(
#     meta_engine,
#     shots_list: List[int] = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100],
#     n_epochs: int = 10,
#     optimizer_factory: Optional[Callable] = None,
#     save_dir: Optional[str] = None,
#     use_wandb: bool = False,
# ):
#     """
#     Convenience function to run adaptation testing.

#     Args:
#         meta_engine: Trained MetaEngine instance
#         shots_list: List of shot counts to test
#         n_epochs: Number of adaptation epochs (single value)
#         optimizer_factory: Factory for adaptation optimizer (default: Adam with lr=0.001)
#         save_dir: Directory to save results
#         use_wandb: Whether to log to wandb

#     Returns:
#         Dictionary mapping subject_id -> DataFrame of results
#     """
#     if optimizer_factory is None:
#         # Default optimizer factory for adaptation
#         def default_optimizer_factory(params):
#             return torch.optim.Adam(params, lr=0.001)

#         optimizer_factory = default_optimizer_factory

#     tester = Test(
#         meta_engine=meta_engine,
#         optimizer_factory=optimizer_factory,
#         use_wandb=use_wandb,
#     )

#     return tester.test_all_subjects(
#         shots_list=shots_list,
#         n_epochs=n_epochs,
#         save_dir=save_dir,
#     )
