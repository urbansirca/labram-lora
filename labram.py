from torcheeg.models.transformer.labram import LaBraM
import torch
from collections import OrderedDict
import pathlib
import torch.nn as nn


# Load the model


def load_labram():
    model = LaBraM.base_patch200_200(num_classes=2)

    # Load checkpoint manually with weights_only=False (since we trust the source)
    MODEL_PATH = (
        pathlib.Path(__file__).parent
        / "weights"
        / "pretrained-models"
        / "labram-base.pth"
    )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False) #TODO: check if head weights are loaded as well

    # Load the state dict into the model
    model.load_state_dict(checkpoint, strict=False)

    print("Original model:")
    print(model)

    # # count number of parameters
    # print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")

    # # Modify the head for 2-class classification
    # model.head = nn.Linear(in_features=200, out_features=2, bias=True)

    # # randomly initialize the head weights with xavier_uniform
    # model.head.weight.data = torch.nn.init.xavier_uniform_(model.head.weight.data)
    # model.head.bias.data = torch.nn.init.zeros_(model.head.bias.data)

    # print("\nModified model for 2-class classification:")
    # print(model)

    # # count number of parameters after modification
    # print(f"Modified model parameters: {sum(p.numel() for p in model.parameters())}")

    return model
