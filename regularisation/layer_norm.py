import torch
import torch.nn as nn


class LayerNorm1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.beta = torch.zeros([self.dim])
        self.gamma = torch.ones([self.dim])

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        return 1 / D * x.sum(dim=0)

    def _variance(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        return 1 / D * torch.pow(x - self._mean(x).unsqueeze(0), 2).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._mean(x)
        variance = self._variance(x)
        epsilon = torch.tensor(1e-7)

        z = x - mean.unsqueeze(0) / torch.sqrt(variance + epsilon).unsqueeze(0)
        y = z * self.gamma.unsqueeze(0) + self.beta.unsqueeze(0)
        return y


if __name__ == "__main__":
    B = 8
    D = 3

    x = torch.rand([B, D])
    ln = LayerNorm1d(D)

    print(x)
    print()
    print(ln(x))
