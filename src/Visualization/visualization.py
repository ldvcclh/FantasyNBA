import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Visualization:
    def __init__(self):
        """
        Initialize the Visualization class.
        """
        pass

    def plot_predicted_vs_actual(self, models_results, player_name=None):
        """
        Plot Predicted vs Actual values for one or multiple players.

        Args:
        - models_results (dict): Dictionary where keys are player names and values are dicts with 'actual' and 'predicted'.
        - player_name (str): Specific player name to plot. If None, plots all players on the same graph.
        """
        plt.figure(figsize=(10, 6))
        for player, results in models_results.items():
            if player_name and player != player_name:
                continue
            plt.scatter(results['actual'], results['predicted'], label=player, alpha=0.7)

        plt.plot([min(results['actual']), max(results['actual'])],
                 [min(results['actual']), max(results['actual'])],
                 color='red', linestyle='--', label='Ideal Fit')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_residual_distribution(self, models_results, player_name=None):
        """
        Plot the distribution of residuals for one or multiple players.

        Args:
        - models_results (dict): Dictionary where keys are player names and values are dicts with 'actual' and 'predicted'.
        - player_name (str): Specific player name to plot. If None, plots all players on the same graph.
        """
        plt.figure(figsize=(10, 6))
        for player, results in models_results.items():
            if player_name and player != player_name:
                continue
            residuals = np.array(results['actual']) - np.array(results['predicted'])
            sns.kdeplot(residuals, label=f'{player}', fill=True, alpha=0.6)

        plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Residual Distribution')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_model_comparison(self, models_metrics, metric_name):
        """
        Compare models based on a specific metric.

        Args:
        - models_metrics (dict): Dictionary where keys are model names and values are metrics (e.g., RMSE, R2).
        - metric_name (str): The name of the metric to display.
        """
        plt.figure(figsize=(10, 6))
        model_names = list(models_metrics.keys())
        metric_values = list(models_metrics.values())

        sns.barplot(x=model_names, y=metric_values)
        plt.xlabel('Models')
        plt.ylabel(metric_name)
        plt.title(f'Model Comparison based on {metric_name}')
        plt.grid()
        plt.show()