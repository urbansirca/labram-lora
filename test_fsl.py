import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import pandas as pd
from meta_helpers import (
    sample_support,
    sample_query,
    fetch_by_indices,
)

logger = logging.getLogger(__name__)


class Test:
    def __init__(
        self,
        meta_engine,
        optimizer_factory: Optional[Callable[[List], torch.optim.Optimizer]] = None,
        use_wandb: bool = False,
        wandb_prefix: str = "test",
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

    def _zero_shot_evaluate(self, subject_id: int) -> Dict[str, float]:
        """Evaluate model on subject without any adaptation."""
        self.meta_engine.model.eval()
        
        # Get all available trials for this subject as query
        available_trials = list(range(len(self.meta_engine.test_epi[subject_id])))
        
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
            
        return {
            "accuracy": accuracy.item(),
            "loss": loss.item(),
            "n_samples": len(yq)
        }

    def _adapt_and_evaluate(
        self, 
        subject_id: int, 
        n_shots: int, 
        n_epochs: int
    ) -> Dict[str, float]:
        """Adapt model on n_shots and evaluate on remaining trials."""
        self.meta_engine.model.eval()
        
        # Sample support and query indices
        sup_idx, remaining_trials = sample_support(
            subject_id, self.meta_engine.test_epi, n_shots, self.rng
        )
        que_idx = sample_query(
            subject_id, remaining_trials, self.meta_engine.test_epi, None, self.rng
        )
        
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
        
        # Adaptation loop
        for epoch in range(n_epochs):
            adapter_optimizer.zero_grad()
            
            # Create parameter dict for functional call
            fast_dict = dict(zip(base_names, fast))
            
            # Forward pass on support set
            logits_s = self.meta_engine._forward_with(fast_dict, Xs)
            loss_s = nn.functional.cross_entropy(logits_s, ys)
            
            # Backward pass
            loss_s.backward()
            adapter_optimizer.step()
        
        # Evaluate on query set
        fast_dict = dict(zip(base_names, fast))
        with torch.no_grad():
            logits_q = self.meta_engine._forward_with(fast_dict, Xq)
            loss_q = nn.functional.cross_entropy(logits_q, yq)
            accuracy = (logits_q.argmax(1) == yq).float().mean()
        
        return {
            "accuracy": accuracy.item(),
            "loss": loss_q.item(),
            "n_support": len(ys),
            "n_query": len(yq)
        }

    def test_subject_adaptation(
        self,
        subject_id: int,
        shots_list: List[int] = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100],
        epochs_list: List[int] = [1, 2, 5, 10, 20],
    ) -> pd.DataFrame:
        """
        Test adaptation for a single subject across different shots and epochs.
        
        Returns:
            DataFrame with shots as rows, epochs as columns, accuracy as values
        """
        logger.info(f"Testing subject {subject_id}")
        
        results = {}
        
        for n_shots in shots_list:
            if n_shots == 0:
                # Zero-shot evaluation
                result = self._zero_shot_evaluate(subject_id)
                accuracy = result["accuracy"]
                # For zero-shot, all epoch columns have same value
                for n_epochs in epochs_list:
                    results[(n_shots, n_epochs)] = accuracy
            else:
                # Check if we have enough trials for this many shots
                available_trials = len(self.meta_engine.test_epi[subject_id])
                if n_shots >= available_trials:
                    logger.warning(
                        f"Subject {subject_id}: Requested {n_shots} shots but only "
                        f"{available_trials} trials available. Skipping."
                    )
                    for n_epochs in epochs_list:
                        results[(n_shots, n_epochs)] = float('nan')
                    continue
                
                for n_epochs in epochs_list:
                    result = self._adapt_and_evaluate(subject_id, n_shots, n_epochs)
                    results[(n_shots, n_epochs)] = result["accuracy"]
        
        # Convert to DataFrame
        df = pd.DataFrame(index=shots_list, columns=epochs_list)
        for (shots, epochs), accuracy in results.items():
            df.loc[shots, epochs] = accuracy
            
        return df

    def test_all_subjects(
        self,
        shots_list: List[int] = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100],
        epochs_list: List[int] = [1, 2, 5, 10, 20],
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Test adaptation for all test subjects.
        
        Returns:
            Dictionary mapping subject_id -> DataFrame of results
        """
        logger.info(f"Testing {len(self.meta_engine.S_test)} subjects")
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        aggregated_results = []
        
        for subject_id in self.meta_engine.S_test:
            start_time = time.time()
            
            # Test this subject
            subject_df = self.test_subject_adaptation(subject_id, shots_list, epochs_list)
            all_results[subject_id] = subject_df
            
            # Save individual subject results
            if save_dir:
                subject_results = {
                    "subject_id": subject_id,
                    "results": subject_df.to_dict(),
                    "shots_list": shots_list,
                    "epochs_list": epochs_list,
                    "timestamp": time.time(),
                }
                
                with open(save_dir / f"subject_{subject_id}_results.json", "w") as f:
                    json.dump(subject_results, f, indent=2)
            
            # Log to wandb if requested
            if self.use_wandb and self.meta_engine.wandb_run is not None:
                # Log individual subject metrics
                for shots in shots_list:
                    for epochs in epochs_list:
                        accuracy = subject_df.loc[shots, epochs]
                        if not pd.isna(accuracy):
                            self.meta_engine.wandb_run.log({
                                f"{self.wandb_prefix}/subject_{subject_id}/shots_{shots}/epochs_{epochs}/accuracy": accuracy
                            })
            
            # Collect for aggregation
            for shots in shots_list:
                for epochs in epochs_list:
                    accuracy = subject_df.loc[shots, epochs]
                    if not pd.isna(accuracy):
                        aggregated_results.append({
                            "subject_id": subject_id,
                            "shots": shots,
                            "epochs": epochs,
                            "accuracy": accuracy
                        })
            
            elapsed = time.time() - start_time
            logger.info(f"Subject {subject_id} completed in {elapsed:.2f}s")
        
        # Log aggregated results to wandb
        if self.use_wandb and self.meta_engine.wandb_run is not None:
            agg_df = pd.DataFrame(aggregated_results)
            
            for shots in shots_list:
                for epochs in epochs_list:
                    subset = agg_df[(agg_df['shots'] == shots) & (agg_df['epochs'] == epochs)]
                    if not subset.empty:
                        mean_acc = subset['accuracy'].mean()
                        std_acc = subset['accuracy'].std()
                        self.meta_engine.wandb_run.log({
                            f"{self.wandb_prefix}/mean_accuracy/shots_{shots}/epochs_{epochs}": mean_acc,
                            f"{self.wandb_prefix}/std_accuracy/shots_{shots}/epochs_{epochs}": std_acc,
                        })
        
        # Save aggregated results
        if save_dir:
            summary = {
                "experiment_name": self.meta_engine.experiment_name,
                "test_subjects": self.meta_engine.S_test,
                "shots_list": shots_list,
                "epochs_list": epochs_list,
                "aggregated_results": aggregated_results,
                "timestamp": time.time(),
            }
            
            with open(save_dir / "aggregated_results.json", "w") as f:
                json.dump(summary, f, indent=2)
        
        logger.info(f"Testing completed for all {len(self.meta_engine.S_test)} subjects")
        return all_results


# Example usage function
def run_adaptation_test(
    meta_engine,
    shots_list: List[int] = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100],
    epochs_list: List[int] = [1, 2, 5, 10, 20],
    optimizer_factory: Optional[Callable] = None,
    save_dir: Optional[str] = None,
    use_wandb: bool = False,
):
    """
    Convenience function to run adaptation testing.
    
    Args:
        meta_engine: Trained MetaEngine instance
        shots_list: List of shot counts to test
        epochs_list: List of adaptation epoch counts to test
        optimizer_factory: Factory for adaptation optimizer (default: Adam with lr=0.001)
        save_dir: Directory to save results
        use_wandb: Whether to log to wandb
    
    Returns:
        Dictionary mapping subject_id -> DataFrame of results
    """
    if optimizer_factory is None:
        # Default optimizer factory for adaptation
        def default_optimizer_factory(params):
            return torch.optim.Adam(params, lr=0.001)
        optimizer_factory = default_optimizer_factory
    
    tester = Test(
        meta_engine=meta_engine,
        optimizer_factory=optimizer_factory,
        use_wandb=use_wandb,
    )
    
    return tester.test_all_subjects(
        shots_list=shots_list,
        epochs_list=epochs_list,
        save_dir=save_dir,
    )