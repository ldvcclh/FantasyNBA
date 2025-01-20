import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error


class Model:
    def __init__(self, features, output, log=False):
        """
        Initialize the Model class with features and output.
        """
        self.features = features
        self.output = output
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_pred = None
        self.log = log
        self.models_available = {
            "linear_regression": LinearRegression,
            "random_forest": RandomForestRegressor,
            "svm": SVR
        }

    def create_train_test_split(self, test_size=0.3, random_state=42, shuffle=False):
        """
        Split the data into training and testing sets.
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.output, test_size=test_size, random_state=random_state,shuffle=shuffle)
        if self.log:
            print("Train-test split completed.")

    def build_model(self, model_type: str, **kwargs):
        """
        Build a model based on the specified type.
        """
        if model_type not in self.models_available:
            raise ValueError(
                f"Model type '{model_type}' is not supported. Choose from {list(self.models_available.keys())}.")
        self.model = self.models_available[model_type](**kwargs)
        if self.log:
            print(f"{model_type} model initialized.")

    def train_model(self):
        """
        Train the initialized model on the training data.
        """
        if self.model is None:
            raise ValueError("No model has been initialized. Use build_model() first.")
        self.model.fit(self.x_train, self.y_train)
        if self.log:
            print("Model training completed.")

    def evaluate_model(self):
        """
        Evaluate the model on the test data and return performance metrics.
        """
        if self.model is None:
            raise ValueError("No model has been initialized or trained.")
        self.y_pred = self.model.predict(self.x_test)
        metrics = {
            "MSE": mean_squared_error(self.y_test, self.y_pred),
            "MAE": mean_absolute_error(self.y_test, self.y_pred),
            "RMSE": root_mean_squared_error(self.y_test, self.y_pred),
            "R2 Score": r2_score(self.y_test, self.y_pred)
        }
        if self.log:
            print("Model evaluation completed.")
        return metrics

    def predict(self, new_data):
        """
        Predict outputs for new data using the trained model.
        """
        if self.model is None:
            raise ValueError("No model has been initialized or trained.")
        return self.model.predict(new_data)
