import copy

import torch
from torch.utils.data import DataLoader
import torch.nn as nn



def test_few_shot(model: nn.Module, test_loader: DataLoader, n_shots: int, n_epochs: int):

    # copy the model, then split the dataset into n_shots and the rest
    model_copy = copy.deepcopy(model)
    model_copy.eval()


    # split dataloader into n_shots and the rest
    test_loader_copy = copy.deepcopy(test_loader)

    # show the shape of the dataloader
    print(test_loader_copy.dataset.shape)

    print(test_loader_copy)

