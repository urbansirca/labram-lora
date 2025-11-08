import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset

@dataclass
class SplitConfig:
    subject_ids: List[int]
    m_leave_out: Optional[int] = None
    subject_ids_leave_out: Optional[List[int]] = None
    train_proportion: float = 0.9
    seed: int = 111

    def __post_init__(self):
        if (self.subject_ids_leave_out is None) == (self.m_leave_out is None):
            print(self.subject_ids_leave_out, self.m_leave_out)
            raise ValueError(
                "Specify exactly one of subject_ids_leave_out OR m_leave_out."
            )
        if not (0.0 < self.train_proportion < 1.0):
            raise ValueError("train_proportion should be a number between 0 and 1.")


class SplitManager:
    def __init__(self, cfg: SplitConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.S = list(cfg.subject_ids)
        self.rng.shuffle(self.S)

        # Build outer folds as lists of (train_subjects, test_subjects)
        self.S_train_pool, self.S_test = self._build_train_test_split()
        self.S_train, self.S_val = self._build_train_validation_split()

    def _build_train_test_split(self) -> Tuple[List[int], List[int]]:
        if self.cfg.m_leave_out is not None:
            M = self.cfg.m_leave_out
            S_test = self.rng.sample(self.S, M)
        else:
            S_test = list(self.cfg.subject_ids_leave_out)

        S_train_pool = list(set(self.S) - set(S_test))
        return S_train_pool, S_test

    def _build_train_validation_split(self) -> Tuple[List[int], List[int]]:
        n_train = math.floor(len(self.S_train_pool) * self.cfg.train_proportion)
        S_train = self.rng.sample(self.S_train_pool, n_train)
        S_val = list(set(self.S_train_pool) - set(S_train))

        return S_train, S_val


class KUTrialDataset(Dataset):
    """
    Optimized version with fast initialization
    """

    def __init__(
        self,
        dataset_path: str,
        subject_ids: List[int],
        as_float32: bool = True,
    ):
        self.dataset_path = str(dataset_path)
        self.subject_ids = list(subject_ids)
        self.as_float32 = as_float32
        self._file = None

        # Build index: [(sid, trial_idx), ...]
        with h5py.File(self.dataset_path, "r") as f:
            self._index: List[Tuple[int, int]] = []

            # Optimized: Assume all subjects have same number of trials
            # Check first subject to get trial count, then apply to all
            first_sid = self.subject_ids[0]
            first_grp = f[f"s{first_sid}"]
            n_trials_per_subject = first_grp["X"].shape[0]

            X_shape = first_grp["X"].shape
            Y_shape = first_grp["Y"].shape
            
            # print shapes for all subjects and check consistency
            inconsistent = []
            for sid in self.subject_ids:
                grp = f[f"s{sid}"]
                if grp["X"].shape != X_shape or grp["Y"].shape != Y_shape:
                    print(f"Warning: Subject {sid} has different data shape: X={grp['X'].shape}, Y={grp['Y'].shape}")
                    inconsistent.append(sid)
            # Fast index building - no need to access each subject's HDF5 group
            for sid in self.subject_ids:
                self._index.extend((sid, i) for i in range(n_trials_per_subject) if sid not in inconsistent)

            total_trials = len(self._index)

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")
        return self._file

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int):
        import numpy as np

        sid, t = self._index[idx]
        grp = self.file[f"s{sid}"]

        X_np = grp["X"][t]  # (C, P, T)
        Y_np = grp["Y"][t]  # scalar
        
        X = (
            torch.from_numpy(X_np).float()
            if X_np.dtype != np.float32
            else torch.from_numpy(X_np)
        )
        Y = torch.as_tensor(Y_np, dtype=torch.long)

        return X, Y, sid

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None