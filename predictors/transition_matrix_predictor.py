# predictors/transition_matrix_predictor.py

import numpy as np
from tqdm import tqdm


class TransitionMatrixPredictor:
    def __init__(self, deck_size):
        self.deck_size = deck_size
        self.transition_matrix = np.zeros((deck_size, deck_size))
        self.orders = []
        self.current_matrix = None

    def train(self, shuffle_function, num_shuffles=10000):
        print("Training the Transition Matrix model...")
        for _ in tqdm(range(num_shuffles)):
            deck = shuffle_function(np.arange(self.deck_size))
            for i in range(len(deck)):
                self.transition_matrix[deck[i], i] += 1
            self.orders.append([deck.index(i) for i in range(len(deck))])

        self.transition_matrix = self.transition_matrix / num_shuffles
        self.current_matrix = self.transition_matrix.copy()

    def predict_probabilities(self, dealt_cards):
        for i, card in enumerate(dealt_cards):
            # zeroing out impossible
            self.current_matrix[card][:] = 0
        return self.current_matrix.T[len(dealt_cards)]

    def reset(self):
        # Reset the current probabilities to the initial probabilities
        self.current_matrix = self.transition_matrix.copy()
