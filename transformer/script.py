import torch
from bigram import BigramLanguageModel

# Hyperparameters
torch.manual_seed(1337)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERATIONS = 5000
LEARNING_RATE = 1e-3
EVAL_EVERY = 100


def load_dataset(path: str) -> str:
    with open(path, 'r') as f:
        text = f.read()
    return text


def get_vocab(text: str) -> list:
    return sorted(list(set(text)))


def get_batch(split: str):
    assert split in ['train', 'val']
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


dataset = load_dataset('datasets/tiny-shakespeare.txt')
vocab = get_vocab(dataset)

stoi = {u:i for i, u in enumerate(vocab)}
itos = {i:u for i, u in enumerate(vocab)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])


data = torch.tensor(encode(dataset), dtype=torch.long).to(DEVICE)
train_val_ratio = int(0.9 * len(data))
train_data = data[:train_val_ratio]
val_data = data[train_val_ratio:]

model = BigramLanguageModel(len(vocab))
model.to(DEVICE)

optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_EVERY)
        for k in range(100):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


print("Untrained model:")
input_idx = torch.zeros(1, 1, dtype=torch.long, device=DEVICE) # start with a single token {0: '\n'}
print(decode(model.generate(input_idx, max_new_tokens=100)[0].tolist()))



for iters in range(MAX_ITERATIONS):
    if iters % EVAL_EVERY == 0:
        losses = estimate_loss()
        print(f"Step {iters}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()


input_idx = torch.zeros(1, 1, dtype=torch.long, device=DEVICE) # start with a single token {0: '\n'}
print(decode(model.generate(input_idx, max_new_tokens=100)[0].tolist()))
