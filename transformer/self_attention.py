import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


B,T,C = 4, 8, 32 # batch, time, channels
x = torch.randn(B,T,C)

# Let's see a single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B,T,head_size)
q = query(x) # (B,T,head_size)

# Note that no communication between k and q has taken place yet
wei = q @ k.transpose(-2,-1) # (B,T, 16) @ (B,16,T) = (B,T,T)

# Scale the weights
# wei = q @ k.transpose(-2,-1) / (head_size**0.5) # (B,T,T)

print(wei.var())

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

print(out.shape)