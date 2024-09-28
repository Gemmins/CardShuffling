# games/odd_even_game.py

import numpy as np
from tqdm import tqdm


class OddEvenGame:
    def __init__(self, deck_size):
        self.deck_size = deck_size

    def play(self, shuffle_function, predictor, num_games):
        predictor_correct = 0
        standard_correct = 0


        for _ in tqdm(range(num_games)):
            deck = shuffle_function(np.arange(self.deck_size))
            dealt_cards = []

            for i in range(self.deck_size - 1):
                next_card = deck[i]

                probabilities = predictor.predict_probabilities(dealt_cards)
                a = np.sum(probabilities[1::2])
                b = np.sum(probabilities[0::2])
                predictor_prediction = "odd" if a > b else "even"
                standard_prediction = self.standard_predict(dealt_cards)

                actual = "odd" if next_card % 2 == 1 else "even"

                if predictor_prediction == actual:
                    predictor_correct += 1
                if standard_prediction == actual:
                    standard_correct += 1

                dealt_cards.append(deck[i])

            predictor.reset()


        predictor_accuracy = predictor_correct / (num_games * (self.deck_size - 1))
        standard_accuracy = standard_correct / (num_games * (self.deck_size - 1))

        print(f"Predictor Accuracy: {predictor_accuracy:.2%}")
        print(f"Standard Prediction Accuracy: {standard_accuracy:.2%}")

        return predictor_accuracy, standard_accuracy

    def standard_predict(self, dealt_cards):
        remaining_cards = set(range(self.deck_size)) - set(dealt_cards)
        odd_count = sum(1 for card in remaining_cards if card % 2 == 1)
        even_count = sum(1 for card in remaining_cards if card % 2 == 0)
        return "odd" if odd_count > even_count else "even"