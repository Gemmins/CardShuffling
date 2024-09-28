# predictors/transition_matrix_predictor.py

import numpy as np
from tqdm import tqdm

class TransitionMatrixPredictor:
    def __init__(self, deck_size):
        self.deck_size = deck_size
        self.transition_matrix = np.zeros((deck_size, deck_size))
        self.initial_probabilities = None
        self.current_probabilities = None

    def train(self, shuffle_function, num_shuffles=10000):
        print("Training the Transition Matrix model...")
        for _ in tqdm(range(num_shuffles)):
            deck = shuffle_function(np.arange(self.deck_size))
            for i in range(len(deck) - 1):
                self.transition_matrix[deck[i], deck[i+1]] += 1

        # Normalize the transition matrix
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]

        # Calculate initial probabilities based on the first card of each shuffle
        initial_counts = np.zeros(self.deck_size)
        for _ in range(num_shuffles):
            deck = shuffle_function(np.arange(self.deck_size))
            initial_counts[deck[0]] += 1
        self.initial_probabilities = initial_counts / num_shuffles

        self.reset()

    def predict_probabilities(self, dealt_cards):
        if not dealt_cards:
            return self.current_probabilities

        last_card = dealt_cards[-1]
        self.current_probabilities = self.transition_matrix[last_card]

        # Zero out probabilities for cards already dealt
        for card in dealt_cards:
            self.current_probabilities[card] = 0

        # Renormalize
        total_prob = self.current_probabilities.sum()
        if total_prob > 0:
            self.current_probabilities /= total_prob
        else:
            # If all probabilities are zero, return uniform distribution over remaining cards
            remaining_cards = set(range(self.deck_size)) - set(dealt_cards)
            self.current_probabilities[list(remaining_cards)] = 1 / len(remaining_cards)

        return self.current_probabilities

    def reset(self):
        # Reset the current probabilities to the initial probabilities
        self.current_probabilities = self.initial_probabilities.copy()