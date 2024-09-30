# predictors/conditional_predictor.py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class ConditionalPredictor:
    def __init__(self, deck_size):
        self.current_matrix = None
        self.deck_size = deck_size
        self.transition_matrix = np.zeros((deck_size, deck_size))
        self.orders = []
        self.cond_probs = np.zeros((self.deck_size, self.deck_size, self.deck_size, self.deck_size))


    def train(self, shuffle_function, num_shuffles=10000):
        print("Training the Conditional model...")
        for _ in tqdm(range(num_shuffles)):
            deck = shuffle_function(np.arange(self.deck_size))
            for i in range(len(deck)):
                self.transition_matrix[deck[i], i] += 1
            self.orders.append([deck.index(i) for i in range(len(deck))])
        self.orders = np.array(self.orders)
        self.cond_probs = np.zeros((self.deck_size, self.deck_size, self.deck_size, self.deck_size))

        for card_n in tqdm(range(self.deck_size), desc="Computing matrices"):
            for position_m in range(self.deck_size):
                # Filter orders where card_n is at position_m
                mask = self.orders[:, position_m] == card_n
                filtered_orders = self.orders[mask]

        self.transition_matrix = self.transition_matrix / num_shuffles

    def predict_probabilities(self, dealt_cards):
        # after each new card is dealt, update the current matrix to predict the next card
        self.current_matrix = self.transition_matrix.copy()
        if len(dealt_cards) > 0:
            self.current_matrix = np.zeros((self.deck_size, self.deck_size))
            mask = self.orders[:, len(dealt_cards)-1] == dealt_cards[-1]
            filtered_orders = self.orders[mask]
            if len(filtered_orders) > 0:
                for order in filtered_orders:
                    for i in range(len(order)):
                        self.current_matrix[order[i], i] += 1
                self.current_matrix = self.current_matrix / len(filtered_orders)
            else:
                self.current_matrix = self.transition_matrix
        for card in dealt_cards:
            # zeroing out impossible
            self.current_matrix[card][:] = 0
        return self.current_matrix.T[len(dealt_cards)]


    def reset(self):
        # Reset the current probabilities to the initial probabilities
        self.current_matrix = self.transition_matrix.copy()