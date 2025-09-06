import torch
from torch import nn
from functools import reduce
from operator import __add__

class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)
    

class EEGNet(nn.Module):

    def __init__(self, nb_classes, Chans = 22, Samples = 1001,
                 dropoutRate = 0.5, kernLength = 64, F1 = 8,
                 D = 2, F2 = 16, norm_rate = 0.25, device = "cpu") -> None:
        super().__init__()
        self.device = device

        self.block1 = nn.Sequential(
            Conv2dSamePadding(1, F1, (1, kernLength), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        ).to(device)

        self.block2 = nn.Sequential(
            Conv2dSamePadding(F2, F2, (1, 16), bias=False),
            nn.Conv2d(F2, F2, 1, padding=0, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        ).to(device)

        self.block_weights = ['block1.0.weight', 'block1.1.weight', 'block1.1.bias', 'block1.2.weight', 'block1.3.weight', 'block1.3.bias', 'block2.0.weight', 'block2.1.weight', 'block2.2.weight', 'block2.2.bias']

        self.classifier_input = 16 * ((Samples // 4) // 8)
        self.classifier_hidden = int((self.classifier_input * nb_classes) ** 0.5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.classifier_input, self.classifier_hidden),
            nn.Linear(self.classifier_hidden, nb_classes)
        ).to(device)

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x, target=None):
        x = self.block1(torch.unsqueeze(x, 1))
        x = self.block2(x)
        pred = self.classifier(x)

        loss = self.criterion(pred, target)
        return loss, pred

    def freeze(self):
        for name, param in self.named_parameters():
            if name in self.block_weights:
                param.requires_grad = False

    def unfreeze(self):
        for name, param in self.named_parameters():
            if name in self.block_weights:
                param.requires_grad = True