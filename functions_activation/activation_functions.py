import math
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def relu(x: torch.Tensor) -> torch.Tensor:
    mask = x > 0
    return x * mask


def leaky_relu(x: torch.Tensor, slope: Optional[float] = 0.1) -> torch.Tensor:
    mask = x > 0
    return mask * x + (~mask) * slope * x


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1 + torch.erf(x / torch.sqrt(torch.Tensor([2.0]))))


def tanh(x: torch.Tensor) -> torch.Tensor:
    return (math.e ** (2 * x) - 1) / (math.e ** (2 * x) + 1)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + math.e ** (-x))


if __name__ == "__main__":
    x = torch.arange(-20, 20, 0.1)

    print("x", x)
    print("Activation Functions")
    print("relu(x)", relu(x))
    print("leaky_relu(x, 0.1)", leaky_relu(x))
    print("gelu(x)", gelu(x))
    print("tanh(x)", tanh(x))
    print("sigmoid(x)", sigmoid(x))

    # Create a figure for 5 subplots
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))

    fig.suptitle("Activation Functions", fontsize=20)

    fig.text(
        0.5,
        0.04,
        "Input Signal",
        ha="center",
        fontsize=16,
    )

    fig.text(
        0.04,
        0.5,
        "Activation Signal",
        va="center",
        rotation="vertical",
        fontsize=16,
    )

    # Plot the functions
    ax[0].plot(x.numpy())
    ax[0].set_title("Linear")
    ax[1].plot(x.numpy(), relu(x).numpy())
    ax[1].set_title("ReLU")
    ax[2].plot(x.numpy(), leaky_relu(x).numpy())
    ax[2].set_title("Leaky ReLU")
    ax[3].plot(x.numpy(), gelu(x).numpy())
    ax[3].set_title("GELU")
    ax[4].plot(x.numpy(), tanh(x).numpy())
    ax[4].set_title("Tanh")
    ax[5].plot(x.numpy(), sigmoid(x).numpy())
    ax[5].set_title("Sigmoid")

    # save the figure
    plt.savefig("functions_activation/activation_functions.png")
