import torch
from torch import nn
import pandas as pd
from collections import Counter
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Set device to: {DEVICE}")


class RedditCleanJokes(torch.utils.data.Dataset):
    def __init__(
        self,
        file_location: str,
        sequence_length: int,
    ):
        self.file_location = file_location
        self.sequence_length = sequence_length
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]
    
    def load_words(self):
        train_df = pd.read_csv(self.file_location)
        text = train_df['Joke'].str.cat(sep=' ')
        return text.split(' ')
    
    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
    def __len__(self):
        return len(self.words_indexes) - self.sequence_length
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]).to(DEVICE), # X
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length + 1]).to(DEVICE), # y
        )
    
class Lstm(nn.Module):
    def __init__(self, 
                 lstm_size: int,
                 embedding_dim: int,
                 num_layers: int,
                 vocab_size: int):
        super(Lstm, self).__init__()
        self.lstm_size = lstm_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        
        self.fc = nn.Linear(self.lstm_size, vocab_size)


    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state
    

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(DEVICE),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(DEVICE))


def train(dataset, model, hyperparameters):
    model.train()
    dataloader = DataLoader(dataset, 
                            batch_size=hyperparameters['batch_size'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=hyperparameters['learning_rate'])
    
    for epoch in range(hyperparameters["num_epochs"]):
        state_h, state_c = model.init_state(hyperparameters['sequence_length'])
        
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            
            y_pred_tran = y_pred.transpose(1, 2) #[batch, seq_len, vocab_size] => [batch, vocab_size, seq_len]
            loss = criterion(y_pred_tran, y) # [batch, vocab_size, seq_len] , [batch, seq_len]
            
            state_h = state_h.detach()
            state_c = state_c.detach()
            
            loss.backward()
            optimizer.step()
            
            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
    
    
    return model


def predict(dataset, model, prompt, next_words=10):
    model.eval()
    
    words = prompt.split(' ')
    
    state_h, state_c = model.init_state(len(words))
    
    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(DEVICE)
        
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        last_word_logits = y_pred[0][-1]
        
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])
    return words


if __name__ == '__main__':
    dataset = RedditCleanJokes("/homes/hps01/deep-learning/datasets/reddit-clean-jokes.csv", sequence_length=10)
    model = Lstm(lstm_size=256, 
                embedding_dim=256, 
                num_layers=2, 
                vocab_size=len(dataset.get_uniq_words()))
    model.to(DEVICE)

    hyperparameters = {
        "batch_size": 811,
        "learning_rate": 3e-4,
        "num_epochs": 2,
        "sequence_length": 10
    }

    trained_model = train(dataset, model, hyperparameters)


    prompt = "What do you call a"
    for i in range(10):
        print(predict(dataset, trained_model, prompt, next_words=10))