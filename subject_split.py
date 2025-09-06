import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset, Sampler


@dataclass
class SplitConfig:
    #LOMSO:
    subject_ids: List[int]
    m_leave_out: Optional[int] = None
    subject_ids_leave_out: Optional[List[int]] = None

    #Train/Validation
    train_proportion: float = 0.9

    # Reproducibility
    seed: int = 111

    def __post_init__(self):
        if (self.subject_ids_leave_out is None) == (self.m_leave_out is None):
            raise ValueError("Specify exactly one of subject_ids_leave_out OR m_leave_out.")
        if not (0.0 < self.train_proportion < 1.0):
            raise ValueError("train_proportion should be a number between 0 and 1.")


class SplitManager:
    def __init__(self, cfg: SplitConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.S = list(cfg.subject_ids)
        self.rng.shuffle(self.S)

        # Build outer folds as lists of (train_subjects, test_subjects)
        self.S_train_pool, self.S_test= self._build_train_test_split()
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
        n_train  = math.floor(len(self.S_train_pool)*self.cfg.train_proportion)
        S_train = self.rng.sample(self.S_train_pool, n_train )
        S_val = list(set(self.S_train_pool) - set(S_train))

        return S_train, S_val


class KUTrialDataset(Dataset):
    """
    Flattens (subject, trial) so each __getitem__ returns one trial:
      X[s_trial] -> (C, T), Y[s_trial] -> scalar/label
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
            for sid in self.subject_ids:
                grp = f[f"s{sid}"]
                n_trials = grp["X"].shape[0]  # (N, C, T)
                self._index.extend((sid, i) for i in range(n_trials))

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")
        return self._file

    def __len__(self): return len(self._index)

    def __getitem__(self, idx: int):
        import numpy as np
        sid, t = self._index[idx]
        grp = self.file[f"s{sid}"]

        X_np = grp["X"][t]  # (C, T)
        Y_np = grp["Y"][t]  # scalar

        X = torch.from_numpy(X_np).float() if X_np.dtype != np.float32 else torch.from_numpy(X_np)
        Y = torch.as_tensor(Y_np, dtype=torch.long)

        return X, Y, sid

    def close(self):
        if self._file is not None:
            try: self._file.close()
            finally: self._file = None


class SubjectBatchSampler(Sampler[List[int]]):
    """
    Yields batches that are pure by subject. Subject-major ordering:
    all batches for subject A, then subject B, etc. (optionally shuffled).
    Assumes dataset._index: List[(sid, trial_idx)].
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle_subjects: bool = True,
        shuffle_trials: bool = True,
        drop_last: bool = False, # controls what happens to the final, smaller-than-batch-size chunk.
        seed: int = 42,
        subject_order: List[int] | None = None,     # optional explicit order
    ):
        super().__init__(None)
        self.ds = dataset
        self.bs = batch_size
        self.shuf_s = shuffle_subjects
        self.shuf_t = shuffle_trials
        self.drop_last = drop_last 
        self.rng = random.Random(seed)

        # bucket dataset indices by sid
        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, (sid, _t) in enumerate(self.ds._index):
            buckets[sid].append(idx)
        self.buckets = dict(buckets)

        # choose subject order
        if subject_order is not None:
            self.subjects = [s for s in subject_order if s in self.buckets]
        else:
            self.subjects = list(self.buckets.keys())

    def __iter__(self):
        subjects = list(self.subjects)
        if self.shuf_s:
            self.rng.shuffle(subjects)

        for sid in subjects:
            idxs = list(self.buckets[sid])
            if self.shuf_t:
                self.rng.shuffle(idxs)

            # emit batches for this subject (subject-major)
            for i in range(0, len(idxs), self.bs):
                batch = idxs[i:i + self.bs]
                if len(batch) == self.bs or not self.drop_last:
                    yield batch

    def __len__(self):
        total = 0
        for sid, idxs in self.buckets.items():
            n = len(idxs)
            b, r = divmod(n, self.bs)
            total += b + (0 if self.drop_last or r == 0 else 1)
        return total