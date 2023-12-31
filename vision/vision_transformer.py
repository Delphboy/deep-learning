import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model * 4)
        self.linear_2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, in_features: int, d_k: int, d_v: int) -> None:
        super().__init__()
        self.Q = nn.Linear(in_features, d_k, bias=False)
        self.K = nn.Linear(in_features, d_k, bias=False)
        self.V = nn.Linear(in_features, d_v, bias=False)
        self.root_dk = math.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        return self.softmax((Q @ K.transpose(-2, -1)) / self.root_dk) @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, h: Optional[int] = 8) -> None:
        super().__init__()
        assert d_v // h, "Output dimension should be divisible across heads for concat"

        self.attn_heads = nn.ModuleList(
            [AttentionHead(d_model, d_k, d_v // h) for _ in range(h)]
        )

        self.projection = nn.Linear(d_v, d_model, bias=False)

    def forward(self, sequence):
        res = torch.cat([head(sequence) for head in self.attn_heads], dim=-1)
        res = self.projection(res)
        return res


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, dk: int, dv: int, h: int) -> None:
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, dk, dv, h)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.multi_head_attn(x) + x
        x = self.layer_norm_1(x)
        x = self.ffn(x) + x
        x = self.layer_norm_2(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, patch_size: int, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(b, h * w, c)
        return x


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.positional_encodings = nn.Parameter(
            torch.zeros(max_len, 1, d_model), requires_grad=True
        )

    def forward(self, x):
        pe = self.positional_encodings[: x.shape[0]]
        return x + pe


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, n_hidden: int, n_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, n_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes, bias=True),
        )

    def forward(self, x):
        return self.head(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dk: int,
        dv: int,
        h: int,
        N: int = 6,
        in_channels: int = 1,
        patch_size: int = 7,
        num_classes_to_predict: int = 10,
    ) -> None:
        super().__init__()
        self.patcher = PatchEmbedding(d_model, patch_size, in_channels)
        self.pos_emb = LearnedPositionalEmbeddings(d_model)
        self.cls_token_emb = nn.Parameter(
            torch.randn(1, 1, d_model), requires_grad=True
        )
        self.encoder_blocks = nn.Sequential(
            *[EncoderBlock(d_model, dk, dv, h) for _ in range(N)]
        )
        self.classification_head = ClassificationHead(
            d_model, d_model * 2, num_classes_to_predict
        )

    def forward(self, x: torch.Tensor):
        x = self.patcher(x)
        cls = self.cls_token_emb.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.pos_emb(x)
        x = self.encoder_blocks(x)
        cls_token = x[:, 0, :]
        x = self.classification_head(cls_token)
        return x
