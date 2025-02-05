import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class Model:
    def __init__(self, features, target):
        """
        Initialize the Model class with features and target.

        Args:
            features (pd.DataFrame): Feature dataset.
            target (pd.Series): Target variable.
        """
        self.features = features
        self.target = target
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.model = None
        self.scaler = None
        self.history = None

    def create_train_test_split(self, test_size=0.3, standardize='minmax'):
        """
        Create train-test split with optional data standardization.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            standardize (str): Type of standardization ('minmax' or 'standard').
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=42, shuffle=False
        )

        if standardize == 'minmax':
            self.scaler = MinMaxScaler()
        elif standardize == 'standard':
            self.scaler = StandardScaler()

        if self.scaler:
            self.x_train = self.scaler.fit_transform(self.x_train)
            self.x_test = self.scaler.transform(self.x_test)

    def build_model(self, model_type='linear_regression', **kwargs):
        """
        Build the specified regression model.

        Args:
            model_type (str): Type of model ('linear', 'random_forest', 'svm', 'ann').
            **kwargs: Additional arguments for model initialization.
        """
        if model_type == 'linear_regression':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'svm':
            self.model = SVR(**kwargs)
        elif model_type == 'ann':
            self.model = self.build_ann_model(loss = kwargs.get('loss','mae'),
                                              metrics=kwargs.get('metrics', ['mse']),
                                              )
        elif model_type == 'lstm':
            self.model = self.build_lstm_model(loss = kwargs.get('loss','mae'),
                                              metrics=kwargs.get('metrics', ['mse']),
                                              )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def build_ann_model(self, loss, metrics):
        """
        Build a Sequential ANN model.

        Args:
            loss (str): Loss function use for ANN
            metrics (list): Metrics for model evaluation.

        Returns:
            model (Sequential): Compiled ANN model.
        """
        drop = 0.2
        model = Sequential()
        # Input layer
        model.add(Dense(self.x_train.shape[1], input_dim=self.x_train.shape[1], activation='relu'))
        model.add(Dropout(drop))  # Regularization
        # 1st hidden layer
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(drop))
        # 2nd hidden layer
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(drop))
        # Output layer
        model.add(Dense(1, activation='linear'))
        model.compile(loss=loss, optimizer='adam', metrics=metrics)
        return model

    def build_lstm_model(self, loss, metrics):
        """
        Build a Sequential ANN model.

        Args:
            loss (str): Loss function use for ANN
            metrics (list): Metrics for model evaluation.

        Returns:
            model (Sequential): Compiled ANN model.
        """
        drop = 0.1
        units = 32
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 1, self.x_test.shape[1]))
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 1, self.x_train.shape[1]))
        model = Sequential()
        # input layer
        model.add(LSTM(units, return_sequences=True, input_dim=self.x_train.shape[2], activation='relu'))
        model.add(Dropout(drop))  # regularization
        # 1st hidden layer
        model.add(LSTM(units=units, return_sequences=True, input_dim=self.x_train.shape[2], activation='relu'))
        model.add(Dropout(drop))
        # 2nd hidden layer
        model.add(LSTM(units=units, return_sequences=False, input_dim=self.x_train.shape[2], activation='relu'))
        model.add(Dropout(drop))
        # output layer
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mae', metrics=metrics)
        return model

    def train_model(self,**kwargs):
        """
        Fit the model to the training data.
        """
        if isinstance(self.model, Sequential):
            epochs = kwargs.get('epochs',50)
            batch_size = kwargs.get('batch_size',16)
            self.history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                           validation_data=(self.x_test, self.y_test),verbose=0)
        else:
            self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the model on the test data.

        Returns:
            score (float): Evaluation score.
        """
        if isinstance(self.model, Sequential):
            #score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            #self.y_pred = self.model.predict(self.x_test)
            self.y_pred = self.model.predict(self.x_test)
            mse = np.mean((self.y_pred - self.y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(self.y_pred - self.y_test))
            score = {"MSE": mse, "RMSE": rmse, "MAE": mae}
        else:
            self.y_pred = self.model.predict(self.x_test)
            score = {
                    "MSE": mean_squared_error(self.y_test, self.y_pred),
                    "MAE": mean_absolute_error(self.y_test, self.y_pred),
                    "RMSE": root_mean_squared_error(self.y_test, self.y_pred),
                    "R2 Score": r2_score(self.y_test, self.y_pred)
                }
            #score = self.model.score(self.x_test, self.y_test)
        return score

    def predict(self):
        """
        Generate predictions on the test data.

        Returns:
            predictions (np.array): Predicted values.
        """
        return self.model.predict(self.x_test)

    def get_train_test_data(self):
        """
        Retrieve the train-test split data.

        Returns:
            tuple: (x_train, x_test, y_train, y_test)
        """
        return self.x_train, self.x_test, self.y_train, self.y_test


    #def evaluate_model(self):
    #    """
    #    Evaluate the model on the test data and return performance metrics.
    #    """
    #    if self.model is None:
    #        raise ValueError("No model has been initialized or trained.")
    #    self.y_pred = self.model.predict(self.x_test)
    #    metrics = {
    #        "MSE": mean_squared_error(self.y_test, self.y_pred),
    #        "MAE": mean_absolute_error(self.y_test, self.y_pred),
    #        "RMSE": root_mean_squared_error(self.y_test, self.y_pred),
    #        "R2 Score": r2_score(self.y_test, self.y_pred)
    #    }
    #    if self.log:
    #        print("Model evaluation completed.")
    #    return metrics


