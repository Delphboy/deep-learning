from typing import Optional

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    class_labels: torch.Tensor,
    reduction: Optional[str] = "mean",
):
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "Only mean, sum, and none operations are supported"

    logits += 1e-9

    val = F.one_hot(class_labels) * torch.log(logits)

    if reduction == "mean":
        return -torch.mean(val)
    if reduction == "sum":
        return -torch.sum(val)
    return val


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 24
    vocab_size = 10

    logits = torch.rand([batch_size, sequence_length, vocab_size])
    class_labels = torch.randint(0, vocab_size, [batch_size, sequence_length])
    class_labels[-1, -1] = vocab_size - 1

    print(cross_entropy_loss(logits, class_labels, reduction="mean"))
    print(cross_entropy_loss(logits, class_labels, reduction="sum"))
