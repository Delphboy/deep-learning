import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    def __init__(self, batch_dim: int) -> None:
        super().__init__()
        self.batch_dim = batch_dim
        self.beta = torch.zeros([self.batch_dim])
        self.gamma = torch.ones([self.batch_dim])

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        return 1 / B * x.sum(dim=1)

    def _variance(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        return 1 / B * torch.pow(x - self._mean(x).unsqueeze(1), 2).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._mean(x)
        variance = self._variance(x)
        epsilon = torch.tensor(1e-7)

        z = x - mean.unsqueeze(1) / torch.sqrt(variance + epsilon).unsqueeze(1)
        y = z * self.gamma.unsqueeze(1) + self.beta.unsqueeze(1)
        return y


if __name__ == "__main__":
    B = 8
    D = 3

    x = torch.rand([B, D])
    bn = BatchNorm1d(B)

    print(x)
    print()
    print(bn(x))
