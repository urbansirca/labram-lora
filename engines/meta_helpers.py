from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import numpy as np
import torch


# ----------------------------------------------------------------------------------------------
@dataclass()
class EpisodeIndex:
    by_subj_run: Dict[Tuple[int, int], np.ndarray]  # (sid, run) -> local idxs
    by_subj_run_class: Dict[
        Tuple[int, int, int], np.ndarray
    ]  # (sid, run, cls) -> local idxs
    runs_for_subj: Dict[int, List[int]]  # sid -> [run ids]
    classes_for_subj: Dict[int, np.ndarray]  # sid -> classes
    run_size: int
    n_runs: Dict[int, int]  # sid -> n_runs
    subj_local_to_global: Dict[int, np.ndarray]  # sid -> global idxs
    drop_last: bool


def build_episode_index(
    ds, run_size: int = 100, y_key: str = "Y", x_key: str = "X", drop_last: bool = True
) -> EpisodeIndex:
    f = ds.file

    # Build subj_local_to_global from ds._index (already in dataset order)
    buckets: Dict[int, List[int]] = defaultdict(list)
    for gidx, (sid, _t) in enumerate(ds._index):
        buckets[sid].append(gidx)
    subj_local_to_global = {
        sid: np.asarray(gidxs, dtype=np.int64) for sid, gidxs in buckets.items()
    }

    by_subj_run = {}
    by_subj_run_class = {}
    runs_for_subj = {}
    classes_for_subj = {}
    n_runs = {}

    for sid, gidxs in subj_local_to_global.items():
        grp = f[f"s{sid}"]
        y = np.asarray(grp[y_key][:], dtype=np.int64).ravel()
        n = y.shape[0]

        if drop_last:
            n_used = (n // run_size) * run_size
        else:
            n_used = n

        n_full = (
            (n_used + (run_size - 1)) // run_size if not drop_last else (n // run_size)
        )
        n_runs[sid] = n_full
        runs_for_subj[sid] = list(range(n_full))
        classes_for_subj[sid] = np.unique(y)

        # subject-local indices 0..n-1
        local = np.arange(n, dtype=np.int64)

        for r in range(n_full):
            start = r * run_size
            end = min(start + run_size, n)
            if drop_last and (end - start) < run_size:
                continue  # skip ragged tail
            local_run = local[start:end]
            by_subj_run[(sid, r)] = local_run

            y_run = y[local_run]
            if y_run.size:
                for cls in np.unique(y_run):
                    mask = y_run == cls
                    by_subj_run_class[(sid, r, int(cls))] = local_run[mask]


    return EpisodeIndex(
        by_subj_run=by_subj_run,
        by_subj_run_class=by_subj_run_class,
        runs_for_subj=runs_for_subj,
        classes_for_subj=classes_for_subj,
        run_size=run_size,
        n_runs=n_runs,
        subj_local_to_global=subj_local_to_global,
        drop_last=drop_last,
    )


# ----------------------------------------------------------------------------------------------


def _contiguous(block: List[int], k: int, rng: random.Random) -> List[int]:
    if len(block) <= k:
        return block[:]
    start = rng.randrange(0, len(block) - k + 1)
    return block[start : start + k]


def sample_support(sid: int, epi: EpisodeIndex, K_per_class: int, rng: random.Random):
    # Calculate total support trials needed
    n_classes = len(epi.classes_for_subj[sid])
    total_support_needed = n_classes * K_per_class

    # Get all available runs and their sizes
    available_runs = epi.runs_for_subj[sid].copy()
    rng.shuffle(available_runs)

    # Allocate runs for support and query based on trial counts
    support_runs = []
    query_runs = []
    support_trials_allocated = 0

    for run in available_runs:
        run_size = epi.run_size
        if support_trials_allocated < total_support_needed:
            # Still need more support trials
            support_runs.append(run)
            support_trials_allocated += run_size
        else:
            # Support is full, allocate to query
            query_runs.append(run)

    # Sample support from allocated runs
    sup = []
    for c in epi.classes_for_subj[sid]:
        # pool = all trials of class c from support runs
        pool = []
        for run in support_runs:
            pool.extend(epi.by_subj_run_class.get((sid, run, c), []))
        rng.shuffle(pool)
        # take at most K_per_class samples
        take = _contiguous(pool, min(K_per_class, len(pool)), rng)
        sup.extend(take)


    return sup, query_runs


def sample_query(
    sid: int,
    query_runs: List[int],
    epi: EpisodeIndex,
    Q_per_class: int,
    rng: random.Random,
):
    
    que = []
    for c in epi.classes_for_subj[sid]:
        # pool = all trials of class c from query runs
        pool = []
        for run in query_runs:
            pool.extend(epi.by_subj_run_class.get((sid, run, c), []))
        rng.shuffle(pool)
        take = _contiguous(pool, min(Q_per_class, len(pool)), rng)
        que.extend(take)
    return que


def fetch_by_indices(
    ds, #KUTrialDataset
    epi: EpisodeIndex,
    sid: int,
    local_idxs: List[int],
    device: torch.device,
    non_blocking: bool = True,
):
    import numpy as np

    if len(local_idxs) == 0:
        raise ValueError("fetch_by_indices: empty index list")

    gidxs = epi.subj_local_to_global[sid][np.asarray(local_idxs, dtype=np.int64)]

    Xs, ys = [], []
    for g in gidxs:
        x, y, sid_ = ds[int(g)]
        # Optional sanity check (helpful while wiring things up):
        assert (
            sid_ == sid
        ), f"Subject mismatch: got {sid_}, expected {sid} (global idx {g})"
        Xs.append(x)
        ys.append(y)

    X = torch.stack(Xs, 0).to(device, non_blocking=non_blocking)
    y = torch.stack(ys, 0).to(device, non_blocking=non_blocking)
    return X, y