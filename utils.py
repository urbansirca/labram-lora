import os
import h5py
import numpy as np
import torch


def get_KU_dataset():
    dataset_path = "data/KU/KU_dataset.h5"
    dfile = h5py.File(dataset_path, "r")
    return dfile


def get_folds(n_LOMSO_folds, shuffled_subjects):
    """
    Returns a list of lists, where each list contains the subjects for that fold.
    """
    folds = []
    for fold in range(n_LOMSO_folds):
        folds.append(shuffled_subjects[fold::n_LOMSO_folds])
    return folds


# function that creates training, validation and test sets
def create_sets(dfile, shuffled_subjects, n_LOMSO_folds):
    training_set = []
    validation_set = []
    test_set = []
    folds = get_folds(n_LOMSO_folds, shuffled_subjects)


def get_optimizer_scheduler(optimizer_name, scheduler_name):
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD

    if scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")
    return optimizer, scheduler
