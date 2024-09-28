# predictors/lstm_predictor.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, trange


class DeckDataset(Dataset):
    def __init__(self, shuffle_function, num_shuffles, deck_size):
        self.shuffled_decks = [shuffle_function(np.arange(deck_size)) for _ in trange(num_shuffles)]
        self.deck_size = deck_size

    def __len__(self):
        return len(self.shuffled_decks) * (self.deck_size - 1)

    def __getitem__(self, idx):
        deck_idx = idx // (self.deck_size - 1)
        card_idx = idx % (self.deck_size - 1)
        deck = self.shuffled_decks[deck_idx]

        input_seq = deck[:card_idx + 1]
        target = deck[card_idx + 1]

        padded_input = np.pad(input_seq, (0, self.deck_size - len(input_seq)), 'constant',
                              constant_values=self.deck_size)

        return torch.tensor(padded_input, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class LSTMPredictor(nn.Module):
    def __init__(self, deck_size, embedding_dim, hidden_dim):
        super(LSTMPredictor, self).__init__()
        self.embedding = nn.Embedding(deck_size + 1, embedding_dim, padding_idx=deck_size)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, deck_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

    def reset(self):
        pass

def train_model(model, train_loader, val_loader, num_epochs, patience, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.to(device)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    return model


def predict_next_card(model, dealt_cards, deck_size, device):
    model.eval()
    with torch.no_grad():
        padded_input = np.pad(dealt_cards, (0, deck_size - len(dealt_cards)), 'constant', constant_values=deck_size)
        input_tensor = torch.tensor(padded_input, dtype=torch.long).unsqueeze(0).to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    return probabilities


