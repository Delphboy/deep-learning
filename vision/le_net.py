import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
        )

        self.dense = nn.Sequential(
            nn.Linear(400, 120, bias=False),
            nn.Sigmoid(),
            nn.Linear(120, 84, bias=False),
            nn.Sigmoid(),
            nn.Linear(84, 10, bias=True),
        )

    def forward(self, x: torch.Tensor):
        x = self.cnn(x)
        x = self.dense(x)

        return x
