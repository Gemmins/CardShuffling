# games/next_card_game.py

import numpy as np
from tqdm import tqdm


class NextCardGame:
    def __init__(self, deck_size):
        self.deck_size = deck_size

    def play(self, shuffle_function, predictor, num_games):
        total_predictions = 0
        correct_predictions = 0
        baseline_correct = 0

        for _ in tqdm(range(num_games)):
            deck = shuffle_function(np.arange(self.deck_size))
            dealt_cards = []
            predictor.reset()

            for i in range(self.deck_size - 1):  # Predict all but the last card
                next_card = deck[i]

                probabilities = predictor.predict_probabilities(dealt_cards)
                predicted_card = np.argmax(probabilities)

                if predicted_card == next_card:
                    correct_predictions += 1

                # Baseline: probability of guessing correctly from remaining cards
                baseline_correct += 1 / (self.deck_size - i - 1)

                total_predictions += 1
                dealt_cards.append(deck[i])

        accuracy = correct_predictions / total_predictions
        baseline_accuracy = baseline_correct / total_predictions

        print(f"Next Card Prediction Accuracy: {accuracy:.2%}")
        print(f"Baseline (Random Guess) Accuracy: {baseline_accuracy:.2%}")
        print(f"Improvement over Baseline: {(accuracy - baseline_accuracy) / baseline_accuracy:.2%}")
        print(f"Total correct predictions: {correct_predictions}")
        print(f"Total baseline expected correct predictions: {baseline_correct:.2f}")
        print(f"Total predictions made: {total_predictions}")

        return accuracy, baseline_accuracy