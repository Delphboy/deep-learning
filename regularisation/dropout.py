from typing import Optional

import torch


def dropout(tensor: torch.Tensor, p: Optional[float] = 0.5) -> torch.Tensor:
    dropout_mask = torch.zeros_like(tensor).bernoulli_(1 - p)
    dropped = tensor * dropout_mask
    return dropped * (1 / (1 - p))


if __name__ == "__main__":
    tensor = torch.rand([2, 4])
    print(tensor)
    print()

    print(dropout(tensor, p=0.8))
    print(dropout(tensor, p=0.5))
    print(dropout(tensor, p=0.2))
    print(dropout(tensor, p=0.1))
