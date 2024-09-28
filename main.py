# main.py

import numpy as np
import torch
from torch.utils.data import random_split
from predictors.lstm_predictor import LSTMPredictor, DeckDataset, train_model
from predictors.transition_matrix_predictor import TransitionMatrixPredictor
from games.odd_even_game import OddEvenGame
from games.next_card_game import NextCardGame
from games.baccarat_game import BaccaratGame
import shuffles

def main():
    deck_size = 52
    num_shuffles = 10000
    num_games = 1000
    shuffle = shuffles.custom

    # Initialize games
    odd_even_game = OddEvenGame(deck_size)
    next_card_game = NextCardGame(deck_size)
    baccarat_game = BaccaratGame(deck_size)

    # Initialize and train Transition Matrix predictor
    print("Training Transition Matrix Predictor...")
    transition_matrix_pred = TransitionMatrixPredictor(deck_size)
    transition_matrix_pred.train(shuffle, num_shuffles=num_shuffles)

    # Prepare data for neural network models
    full_dataset = DeckDataset(shuffle, num_shuffles, deck_size)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train LSTM predictor
    print("Training LSTM Predictor...")
    embedding_dim = 32
    hidden_dim = 64
    num_epochs = 1
    patience = 5

    lstm_model = LSTMPredictor(deck_size, embedding_dim, hidden_dim)
    lstm_model = train_model(lstm_model, train_loader, val_loader, num_epochs, patience, device)

    # Dictionary of predictors
    predictors = {
        "LSTM": lstm_model,
        "Transition Matrix": transition_matrix_pred,
    }

    # Play games and print results
    print("\nOdd-Even Game Results:")
    for name, predictor in predictors.items():
        accuracy, _ = odd_even_game.play(shuffle, predictor, num_games)
        print(f"{name} Predictor Accuracy: {accuracy:.2%}")

    """
    print("\nNext Card Prediction Game Results:")
    for name, predictor in predictors.items():
        accuracy, baseline_accuracy = next_card_game.play(shuffle, predictor, num_games)
        print(f"{name} Predictor:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Baseline Accuracy: {baseline_accuracy:.2%}")
        print(f"  Improvement over Baseline: {(accuracy - baseline_accuracy) / baseline_accuracy:.2%}")

    
    print("\nBaccarat Game Results:")
    for name, predictor in predictors.items():
        player_win_rate, banker_win_rate, tie_rate, prediction_accuracy, predictor_ev, base_ev = baccarat_game.play(shuffle, predictor, num_games)
        print(f"{name} Predictor:")
        print(f"  Player Win Rate: {player_win_rate:.2%}")
        print(f"  Banker Win Rate: {banker_win_rate:.2%}")
        print(f"  Tie Rate: {tie_rate:.2%}")
        print(f"  Prediction Accuracy: {prediction_accuracy:.2%}")
        print(f"  Predictor EV: {predictor_ev:.4f}")
        print(f"  Base Baccarat EV: {base_ev:.4f}")
        print(f"  EV Improvement: {predictor_ev - base_ev:.4f}")
    """
if __name__ == "__main__":
    main()