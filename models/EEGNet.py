import torch
import torch.nn as nn
import torch.nn.functional as F
import json, pathlib, torch
from dataclasses import asdict, dataclass

class EEGNet(nn.Module):
    """
    PyTorch reimplementation of EEGNet (Keras version you pasted).
    Expects input (B, C, T). Internally reshapes to (B, 1, C, T).
    Returns logits (no softmax), so use CrossEntropyLoss.

    Args:
        nb_classes: number of classes
        Chans:      number of EEG channels (height)
        Samples:    timesteps per trial (width)
        dropoutRate, kernLength, F1, D, F2: as in EEGNet paper/code.
               By default F2 = F1 * D
    """
    def __init__(
        self,
        nb_classes: int,
        Chans: int = 62,
        Samples: int = 1000,
        dropoutRate: float = 0.5,
        kernLength: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
    ):
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        # Block 1: temporal conv then depthwise spatial conv across channels
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),     # downsample time by /4
            nn.Dropout(dropoutRate),
        )

        # Block 2: separable conv = depthwise (1x16) + pointwise (1x1)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, padding=(0, 8), bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),     # additional /8 (total /32 along time)
            nn.Dropout(dropoutRate),
        )

        # Classifier — lazy so you don't have to precompute the flatten size
        self.classifier = nn.Sequential(
            nn.Flatten(),           # -> (B, F2 * floor(Samples/32))
            nn.LazyLinear(nb_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.classifier(x)
        return x  # logits
    
    def save_pretrained(self, save_directory: str): #TODO: do this properly and also add load_weights ...
        path = pathlib.Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        # 2) weights.bin (or .pt)
        torch.save(self.state_dict(), path / "pytorch_model.bin")