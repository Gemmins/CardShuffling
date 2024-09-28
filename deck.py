# Deck class
import numpy as np


class Deck:

    def __init__(self, size):
        self.size = size
        self.deck = np.array(range(size))

    # Takes a list of steps, and applies them to the cards
    def shuffle(self, steps):
        for step in steps:
            self.deck = step(self.deck)

    def reset(self):
        self.deck = np.array(range(self.size))

    # Just nicer than .
    def get(self):
        return self.deck
