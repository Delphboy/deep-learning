import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, head_size, n_embedding) -> None:
        super().__init__()
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        wei = q @ k.transpose(-2,-1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # This makes it a decoder block
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(N_EMBEDDING, N_EMBEDDING) # Added for skip connections
        self.dropout = nn.Dropout(DROPOUT)
    

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # Multiply by 4 to copy the Transformer paper
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # Projection layer added for skip connections
            nn.Dropout(DROPOUT)
        )


    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.self_attention = MultiHead(n_head, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)
    

    def forward(self, x):
        # NOTE: The x + self.self_attention(x) is the residual connection/skip connection
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x