import torch
import torch.nn.functional as F


def softmax_one_dim(z):
    assert len(z.shape) == 1, "Only supports one dimension tensors (vectors)"
    return torch.exp(z) / torch.sum(torch.exp(z))


def softmax_n_dim(z, dim=-1):
    return torch.exp(z) / torch.sum(torch.exp(z), dim=dim)


if __name__ == "__main__":
    z_one_dim = torch.rand([20])

    softmax_one = softmax_one_dim(z_one_dim)
    assert torch.isclose(softmax_one.sum(), torch.tensor(1.0)), "Doesn't sum to one"
    assert torch.allclose(
        softmax_one, F.softmax(z_one_dim, dim=0)
    ), "Doesn't match pytorch"

    #######################

    z_n_dim = torch.rand([20, 4, 14])
    d = 0
    softmax_n = softmax_n_dim(z_n_dim, d)
    assert torch.isclose(
        softmax_n.sum(d)[0][0], torch.tensor(1.0)
    ), "Doesn't sum to one"
    assert torch.allclose(softmax_n, F.softmax(z_n_dim, dim=d)), "Doesn't match pytorch"
