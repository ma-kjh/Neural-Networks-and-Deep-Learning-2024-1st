import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from dataset import Shakespeare
from model import CharRNN, CharLSTM

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):  # LSTM case
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:  # RNN case
            hidden = hidden.to(device)
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):  # LSTM case
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:  # RNN case
                hidden = hidden.to(device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)
def main():
    ########### Configuration, You can change if you want ############
    seq_length = 30
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.002
    ##################################################################

    dataset = Shakespeare('./data/shakespeare_train.txt')
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_rnn = CharRNN(dataset.vocab_size, embed_size=128, hidden_size=256, num_layers=2).to(device)
    model_lstm = CharLSTM(dataset.vocab_size, embed_size=128, hidden_size=256, num_layers=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=learning_rate)
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=learning_rate)

    train_losses_rnn, val_losses_rnn = [], []
    train_losses_lstm, val_losses_lstm = [], []

    for epoch in range(num_epochs):
        train_loss_rnn = train(model_rnn, train_loader, criterion, optimizer_rnn, device)
        val_loss_rnn = validate(model_rnn, val_loader, criterion, device)
        train_losses_rnn.append(train_loss_rnn)
        val_losses_rnn.append(val_loss_rnn)

        train_loss_lstm = train(model_lstm, train_loader, criterion, optimizer_lstm, device)
        val_loss_lstm = validate(model_lstm, val_loader, criterion, device)
        train_losses_lstm.append(train_loss_lstm)
        val_losses_lstm.append(val_loss_lstm)

        print(f'Epoch [{epoch+1}/{num_epochs}], RNN Train Loss: {train_loss_rnn:.4f}, Val Loss: {val_loss_rnn:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], LSTM Train Loss: {train_loss_lstm:.4f}, Val Loss: {val_loss_lstm:.4f}')

    plt.figure()
    plt.plot(train_losses_rnn, label='Train Loss (RNN)')
    plt.plot(val_losses_rnn, label='Val Loss (RNN)')
    plt.plot(train_losses_lstm, label='Train Loss (LSTM)')
    plt.plot(val_losses_lstm, label='Val Loss (LSTM)')
    plt.legend()
    plt.savefig('loss_plot.png')

    torch.save(model_lstm.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    main()
