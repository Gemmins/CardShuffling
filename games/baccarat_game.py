# games/baccarat_game.py

import numpy as np
from tqdm import tqdm


class BaccaratGame:
    def __init__(self, deck_size=52):
        self.deck_size = deck_size
        self.player_payout = 1  # Player bet pays 1:1
        self.banker_payout = 0.95  # Banker bet pays 1:1 minus 5% commission
        self.tie_payout = 8  # Tie bet pays 8:1

    def play(self, shuffle_function, predictor, num_games):
        player_wins = 0
        banker_wins = 0
        ties = 0
        correct_predictions = 0
        total_ev = 0
        total_bet_amount = num_games  # Assuming 1 unit bet per game

        for _ in tqdm(range(num_games)):
            deck = shuffle_function(np.arange(self.deck_size))
            dealt_cards = []
            predictor.reset()

            # Deal initial cards
            player = [deck[0], deck[2]]
            banker = [deck[1], deck[3]]
            dealt_cards.extend(deck[:4])

            player_score = self.calculate_score(player)
            banker_score = self.calculate_score(banker)

            # Check for natural win
            if player_score >= 8 or banker_score >= 8:
                winner = self.determine_winner(player_score, banker_score)
            else:
                # Player draw
                if player_score <= 5:
                    player.append(deck[4])
                    dealt_cards.append(deck[4])
                    player_score = self.calculate_score(player)

                    # Banker draw
                    if len(player) == 2:
                        if banker_score <= 5:
                            banker.append(deck[5])
                            dealt_cards.append(deck[5])
                    else:
                        if self.banker_should_draw(banker_score, player[2]):
                            banker.append(deck[5])
                            dealt_cards.append(deck[5])
                else:
                    # Banker draw when player stands
                    if banker_score <= 5:
                        banker.append(deck[4])
                        dealt_cards.append(deck[4])

                banker_score = self.calculate_score(banker)
                winner = self.determine_winner(player_score, banker_score)

            # Predict the winner
            probabilities = predictor.predict_probabilities(dealt_cards)
            predicted_winner = self.predict_winner(probabilities)

            if predicted_winner == winner:
                correct_predictions += 1

            # Calculate EV for this game
            if predicted_winner == 'player':
                if winner == 'player':
                    total_ev += self.player_payout
                elif winner == 'banker':
                    total_ev -= 1
            elif predicted_winner == 'banker':
                if winner == 'banker':
                    total_ev += self.banker_payout
                elif winner == 'player':
                    total_ev -= 1
            else:  # predicted tie
                if winner == 'tie':
                    total_ev += self.tie_payout
                else:
                    total_ev -= 1

            if winner == 'player':
                player_wins += 1
            elif winner == 'banker':
                banker_wins += 1
            else:
                ties += 1

        total_games = player_wins + banker_wins + ties
        player_win_rate = player_wins / total_games
        banker_win_rate = banker_wins / total_games
        tie_rate = ties / total_games
        prediction_accuracy = correct_predictions / total_games
        predictor_ev = total_ev / total_bet_amount

        # Calculate base EV of Baccarat
        base_ev = (player_win_rate * self.player_payout +
                   banker_win_rate * self.banker_payout +
                   tie_rate * self.tie_payout - 1)

        print(f"Player Win Rate: {player_win_rate:.2%}")
        print(f"Banker Win Rate: {banker_win_rate:.2%}")
        print(f"Tie Rate: {tie_rate:.2%}")
        print(f"Prediction Accuracy: {prediction_accuracy:.2%}")
        print(f"Predictor EV: {predictor_ev:.4f}")
        print(f"Base Baccarat EV: {base_ev:.4f}")
        print(f"EV Improvement: {predictor_ev - base_ev:.4f}")

        return player_win_rate, banker_win_rate, tie_rate, prediction_accuracy, predictor_ev, base_ev

    def calculate_score(self, hand):
        return sum(min(card % 13 + 1, 10) for card in hand) % 10

    def determine_winner(self, player_score, banker_score):
        if player_score > banker_score:
            return 'player'
        elif banker_score > player_score:
            return 'banker'
        else:
            return 'tie'

    def banker_should_draw(self, banker_score, player_third_card):
        if banker_score <= 2:
            return True
        elif banker_score == 3:
            return player_third_card != 8
        elif banker_score == 4:
            return player_third_card in [2, 3, 4, 5, 6, 7]
        elif banker_score == 5:
            return player_third_card in [4, 5, 6, 7]
        elif banker_score == 6:
            return player_third_card in [6, 7]
        else:
            return False

    def predict_winner(self, probabilities):
        player_prob = sum(probabilities[card] for card in range(0, self.deck_size, 2))
        banker_prob = sum(probabilities[card] for card in range(1, self.deck_size, 2))

        if player_prob > banker_prob:
            return 'player'
        elif banker_prob > player_prob:
            return 'banker'
        else:
            return 'tie'