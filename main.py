# main.py

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from predictors.lstm_predictor import train_lstm_predictor, save_predictor, load_predictor
from predictors.regression_predictors import RandomForestPredictor, GradientBoostingPredictor, LinearRegressionPredictor
from predictors.positional_netowork_predictor import PositionalNetworkPredictor
from games.odd_even_game import OddEvenGame
from games.next_card_game import NextCardGame
from games.baccarat_game import BaccaratGame
import shuffles
from itertools import product
import os
import json
import logging
import sys
import time
import matplotlib.pyplot as plt

def setup_logger(run_dir):
    logger = logging.getLogger('grid_search')
    logger.setLevel(logging.INFO)

    # Create a file handler
    log_file = os.path.join(run_dir, 'run.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def generate_shuffles(shuffle_function, deck_size, num_shuffles, logger):
    logger.info(f"Generating {num_shuffles} shuffles...")
    shuffled_decks = [shuffle_function(np.arange(deck_size)) for _ in range(num_shuffles)]
    logger.info("Shuffle generation complete.")
    return shuffled_decks


def setup_logger(run_dir):
    logger = logging.getLogger('grid_search')
    logger.setLevel(logging.DEBUG)  # Change to DEBUG for more detailed logs

    # Create a file handler
    log_file = os.path.join(run_dir, 'run.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def grid_search(param_grid, shuffled_decks, shuffle_function, deck_size, num_games, logger):
    odd_even_game = OddEvenGame(deck_size)
    next_card_game = NextCardGame(deck_size)

    results = []

    # Deep Learning models grid search
    for params in product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))
        shuffle_name = current_params.pop('shuffle_name')
        model_type = current_params.pop('model_type')

        logger.info(f"\nTraining {model_type} LSTM Predictor with parameters:")
        for key, value in current_params.items():
            logger.info(f"{key}: {value}")

        if model_type == 'single':
            predictor, train_losses, val_losses = train_lstm_predictor(
                shuffled_decks,
                deck_size,
                **current_params
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        run_dir = save_predictor(predictor, f"{shuffle_name}_{model_type}", model_type,
                                 deck_size, train_losses, val_losses, current_params)

        logger.info(f"\n{model_type.capitalize()} LSTM Predictor Results:")
        odd_even_accuracy, standard_odd_even_accuracy = odd_even_game.play(shuffle_function, predictor, num_games)
        next_card_accuracy, baseline_accuracy = next_card_game.play(shuffle_function, predictor, num_games)

        logger.info(f"{model_type.capitalize()} LSTM - Odd-Even Accuracy: {odd_even_accuracy:.3f}")
        logger.info(f"Standard Odd-Even Accuracy: {standard_odd_even_accuracy:.3f}")
        logger.info(f"{model_type.capitalize()} LSTM - Next Card Accuracy: {next_card_accuracy:.3f}")
        logger.info(f"Baseline Next Card Accuracy: {baseline_accuracy:.3f}")

        results.append({
            "hyperparameters": current_params,
            "model_type": model_type,
            "odd_even_accuracy": odd_even_accuracy,
            "standard_odd_even_accuracy": standard_odd_even_accuracy,
            "next_card_accuracy": next_card_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "run_dir": run_dir
        })

    # Regression models
    predictors = [
        #RandomForestPredictor(deck_size),
        #GradientBoostingPredictor(deck_size),
        #LinearRegressionPredictor(deck_size),
        PositionalNetworkPredictor(deck_size, logger=logger)
    ]

    for predictor in predictors:
        logger.info(f"\nTraining {predictor.__class__.__name__}")
        predictor.train(shuffled_decks)

        logger.info(f"\n{predictor.__class__.__name__} Results:")
        odd_even_accuracy, standard_odd_even_accuracy = odd_even_game.play(shuffle_function, predictor, num_games)
        next_card_accuracy, baseline_accuracy = next_card_game.play(shuffle_function, predictor, num_games)

        logger.info(f"{predictor.__class__.__name__} - Odd-Even Accuracy: {odd_even_accuracy:.3f}")
        logger.info(f"Standard Odd-Even Accuracy: {standard_odd_even_accuracy:.3f}")
        logger.info(f"{predictor.__class__.__name__} - Next Card Accuracy: {next_card_accuracy:.3f}")
        logger.info(f"Baseline Next Card Accuracy: {baseline_accuracy:.3f}")

        results.append({
            "model_type": predictor.__class__.__name__,
            "odd_even_accuracy": odd_even_accuracy,
            "standard_odd_even_accuracy": standard_odd_even_accuracy,
            "next_card_accuracy": next_card_accuracy,
            "baseline_accuracy": baseline_accuracy,
        })

    return results


def create_performance_chart(results, grid_search_dir, logger):
    logger.info("Creating performance chart...")

    # Group results by model type
    lstm_results = [r for r in results if r['model_type'] in ['single', 'distribution']]
    regression_results = [r for r in results if r['model_type'] not in ['single', 'distribution']]

    # Create separate charts for LSTM and regression models
    for model_group, group_results in [('LSTM', lstm_results), ('Regression', regression_results)]:
        if not group_results:
            continue

        model_names = []
        odd_even_accuracies = []
        next_card_accuracies = []
        standard_odd_even_accuracies = []
        baseline_accuracies = []

        for result in group_results:
            if model_group == 'LSTM':
                model_name = f"{result['hyperparameters']['embedding_dim']}_{result['hyperparameters']['hidden_dim']}_{result['hyperparameters']['num_layers']}_{result['hyperparameters']['dropout']}"
            else:
                model_name = result['model_type']

            model_names.append(model_name)
            odd_even_accuracies.append(result['odd_even_accuracy'])
            next_card_accuracies.append(result['next_card_accuracy'])
            standard_odd_even_accuracies.append(result['standard_odd_even_accuracy'])
            baseline_accuracies.append(result['baseline_accuracy'])

        # Create bar chart
        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(20, 10))
        rects1 = ax.bar(x - width / 2, odd_even_accuracies, width, label='Model Odd-Even')
        rects2 = ax.bar(x + width / 2, next_card_accuracies, width, label='Model Next Card')

        ax.axhline(y=np.mean(standard_odd_even_accuracies), color='r', linestyle='--',
                   label='Standard Odd-Even Accuracy')
        ax.axhline(y=np.mean(baseline_accuracies), color='g', linestyle='--', label='Baseline Next Card Accuracy')

        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_group} Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90)

        autolabel(rects1)
        autolabel(rects2)

        ax.text(x[-1] + 0.5, np.mean(standard_odd_even_accuracies),
                f'Standard Odd-Even: {np.mean(standard_odd_even_accuracies):.3f}',
                verticalalignment='bottom', horizontalalignment='left', color='r')
        ax.text(x[-1] + 0.5, np.mean(baseline_accuracies), f'Baseline Next Card: {np.mean(baseline_accuracies):.3f}',
                verticalalignment='bottom', horizontalalignment='left', color='g')

        fig.tight_layout()

        chart_path = os.path.join(grid_search_dir, f'performance_chart_{model_group.lower()}.png')
        plt.savefig(chart_path)
        plt.close()

        logger.info(f"{model_group} performance chart saved to {chart_path}")



def main():
    deck_size = 52
    num_shuffles = 100000
    num_games = 100
    shuffle = shuffles.normal
    shuffle_name = "normal"

    # Create a directory for this grid search run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    grid_search_dir = os.path.join('runs', f'grid_search_{timestamp}')
    os.makedirs(grid_search_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(grid_search_dir)

    # Generate shuffles once
    shuffled_decks = generate_shuffles(shuffle, deck_size, num_shuffles, logger)

    # Define the hyperparameter grid
    param_grid = {
        'model_type': [],
        'embedding_dim': [32],
        'hidden_dim': [128],
        'num_epochs': [20],
        'patience': [5],
        'batch_size': [256],
        'num_layers': [4],
        'dropout': [0.2],
        'learning_rate': [0.002],
        'shuffle_name': [shuffle_name]
    }

    results = grid_search(param_grid, shuffled_decks, shuffle, deck_size, num_games, logger)

    # Save overall results
    results_file = os.path.join(grid_search_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Grid search results saved to {results_file}")

    # Create and save performance charts
    create_performance_chart(results, grid_search_dir, logger)

if __name__ == "__main__":
    main()