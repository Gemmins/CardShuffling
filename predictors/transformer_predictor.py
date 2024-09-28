# predictors/transformer_predictor.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, trange

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerPredictor(nn.Module):
    def __init__(self, deck_size, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerPredictor, self).__init__()
        self.embedding = nn.Embedding(deck_size + 1, d_model, padding_idx=deck_size)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, deck_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * np.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output[:, -1, :]

def train_transformer_model(model, train_loader, val_loader, num_epochs, patience, device):
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

def predict_next_card_transformer(model, dealt_cards, deck_size, device):
    model.eval()
    with torch.no_grad():
        padded_input = np.pad(dealt_cards, (0, deck_size - len(dealt_cards)), 'constant', constant_values=deck_size)
        input_tensor = torch.tensor(padded_input, dtype=torch.long).unsqueeze(0).to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    return probabilities

class TransformerWrapper:
    def __init__(self, model, deck_size, device):
        self.model = model
        self.deck_size = deck_size
        self.device = device

    def predict_probabilities(self, dealt_cards):
        return predict_next_card_transformer(self.model, dealt_cards, self.deck_size, self.device)

    def reset(self):
        pass  # Transformer doesn't need resetting