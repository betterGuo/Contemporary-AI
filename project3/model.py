"""A compact CIFAR-10 CNN with one optional, identity-initialized channel gate."""

import torch
from torch import nn


def _conv(in_channels, out_channels, kernel=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ChannelGate(nn.Module):
    """SE-style channel weighting. 2*sigmoid(0)=1 gives a neutral start."""

    def __init__(self, channels=64, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.reduce = nn.Linear(channels, hidden)
        self.expand = nn.Linear(hidden, channels)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        descriptor = x.mean(dim=(2, 3))
        weights = 2 * torch.sigmoid(self.expand(torch.relu(self.reduce(descriptor))))
        return x * weights[:, :, None, None]


class CifarCNN(nn.Module):
    """Sizes for a 32px input: 16x16, 8x8, 4x4, global average, 10 logits."""

    def __init__(self, attention=False):
        super().__init__()
        # Instantiate the entire common trunk and classifier before the gate:
        # resetting torch.manual_seed to the same value gives identical common
        # parameters, regardless of whether the gate is present.
        self.stage1 = nn.Sequential(_conv(3, 32, kernel=4, stride=2), _conv(32, 32))
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), _conv(32, 64), _conv(64, 64))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), _conv(64, 128), _conv(128, 128))
        self.classifier = nn.Linear(128, 10)
        self.gate = ChannelGate(64) if attention else nn.Identity()

    def forward(self, x):
        x = self.stage1(x)
        x = self.gate(self.stage2(x))
        x = self.stage3(x)
        return self.classifier(x.mean(dim=(2, 3)))


def make_model(attention: bool = False) -> CifarCNN:
    return CifarCNN(attention=attention)
