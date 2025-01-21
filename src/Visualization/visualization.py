import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import chain

class Visualization:
    def __init__(self,models_results, model_str):
        """
        Initialize the Visualization class.
        """
        self.model_str = model_str
        self.models_results = models_results

    def plot_predicted_vs_actual(self,player_name=None, threshold=5):
        """
        Plot Predicted vs Actual values using Seaborn with color-coding for good/bad predictions.

        Args:
        - models_results (dict): Dictionary where keys are player names and values are dicts with 'actual' and 'predicted'.
        - player_name (str): Specific player name to plot. If None, plots all players on the same graph.
        - threshold (int): Threshold for a good prediction (default is 5).
        """
        plt.figure(figsize=(10, 6))
        if player_name:
            actual = np.array(self.models_results[player_name]['points']['y_test'])
            predicted = np.array(self.models_results[player_name]['points']['y_pred'])
        else:
            actual = list(chain.from_iterable(data["points"]["y_test"].values for data in self.models_results.values()))
            predicted = list(chain.from_iterable(data["points"]["y_pred"] for data in self.models_results.values()))

            actual = np.array(actual)
            predicted = np.array(predicted)

        actual = actual.ravel()
        predicted = predicted.ravel()

        # Define colors based on the threshold
        colors = ['green' if abs(a - p) <= threshold else 'red' for a, p in zip(actual, predicted)]

        # Create a DataFrame for Seaborn
        df = pd.DataFrame({
            'Actual': actual,
            'Predicted': predicted,
            'Color': colors
        })

        # Scatter plot using Seaborn
        sns.scatterplot(data=df, x='Actual', y='Predicted', hue='Color', palette={'green': 'green', 'red': 'red'},
                        alpha=0.7)
        plt.plot(actual, actual, color='black', lw=2)

        # Calculate the ratio of good predictions
        good_predictions = sum(1 for color in colors if color == 'green')
        total_predictions = len(colors)
        good_prediction_ratio = good_predictions / total_predictions * 100

        plt.ylim((-20, 100))
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual with Threshold')
        plt.title(f'{self.model_str} - Predicted vs Actual (Good Predictions: {good_prediction_ratio:.2f}%)')
        plt.grid()
        plt.show()

    def plot_residual_distribution(self, player_name=None):
        """
        Plot the distribution of residuals for one or multiple players.

        Args:
        - models_results (dict): Dictionary where keys are player names and values are dicts with 'actual' and 'predicted'.
        - player_name (str): Specific player name to plot. If None, plots all players on the same graph.
        """
        plt.figure(figsize=(10, 6))
        for player, results in self.models_results.items():
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