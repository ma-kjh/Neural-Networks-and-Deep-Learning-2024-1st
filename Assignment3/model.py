import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        embedded = self.embed(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output.reshape(-1, self.hidden_size))
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return initial_hidden

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        embedded = self.embed(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output.reshape(-1, self.hidden_size))
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                          torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return initial_hidden
