import os
import warnings
import joblib
import stumpy
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from multiprocessing import cpu_count
from sklearn.metrics import f1_score
from schema.data_schema import TimeStepClassificationSchema
from tqdm import tqdm

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
print(f"Using n_jobs = {n_jobs}")


class TimeStepClassifier:
    """Timeseries Step classifier.

    This class provides a consistent interface that can be used with other
    TimeStepClassifier models.
    """

    MODEL_NAME = "Distance_Profile_Timeseries_classifier"

    def __init__(
        self,
        data_schema: TimeStepClassificationSchema,
        encode_len: int,
        **kwargs,
    ):
        """
        Construct a new Distance Profile TimeStepClassifier.

        Args:
            data_schema (TimeStepClassificationSchema): The schema of the data.
            encode_len (int): Encoding (history) length.
        """
        self.data_schema = data_schema
        self.encode_len = encode_len
        self.kwargs = kwargs
        self._is_trained = False

    def create_windows_for_prediction(self, data: pd.DataFrame):
        """
        Create windows for prediction.

        Args:
            data (pd.DataFrame): The data to create windows for.

        """
        windows = []
        for i in range(0, len(data), self.encode_len):
            windows.append(data[i : i + self.encode_len])

        return windows

    def normalize(self, data: np.ndarray, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
        normalized = (data - mean) / std

        return normalized, mean, std

    def multi_dimensional_mass(self, query_subsequence, time_series) -> np.ndarray:
        """
        Calculate the multi-dimensional matrix profile.

        Args:
            query_subsequence (np.ndarray): The query subsequence.
            time_series (np.ndarray): The time series.

        Returns:
            np.ndarray: The multi-dimensional matrix profile.
        """
        for dim in range(time_series.shape[1]):
            if dim == 0:
                profile = stumpy.core.mass(
                    query_subsequence[:, dim], time_series[:, dim]
                )
            else:
                profile += stumpy.core.mass(
                    query_subsequence[:, dim], time_series[:, dim]
                )
        return profile

    def fit(self, train_data):
        self.train_data = train_data
        self._is_trained = True

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        labels = []
        grouped = data.groupby(self.data_schema.id_col)
        train_series_ids = set(
            self.train_data[self.data_schema.id_col].unique().tolist()
        )
        for id, group in tqdm(grouped, desc="Generating predictions"):
            series_to_be_searched = self.train_data
            if id in train_series_ids:
                series_to_be_searched = self.train_data[
                    self.train_data[self.data_schema.id_col] == id
                ]

            label_col = series_to_be_searched[self.data_schema.target]
            series_to_be_searched = series_to_be_searched.drop(
                columns=[
                    self.data_schema.id_col,
                    self.data_schema.time_col,
                    self.data_schema.target,
                ]
            ).to_numpy()

            group = group.drop(columns=[self.data_schema.id_col])
            group = group.drop(columns=[self.data_schema.time_col]).to_numpy()

            series_to_be_searched, mean, std = self.normalize(series_to_be_searched)
            group, _, _ = self.normalize(group, mean, std)

            windowed_group = self.create_windows_for_prediction(group)
            for i, window in enumerate(windowed_group):
                if window.shape[0] < self.encode_len:
                    if i == 0:
                        raise InsufficientDataError(
                            f"The length of the input data is less than the encode length. Input data length ({window.shape[0]}) < encode length ({self.encode_len})."
                        )

                    last_pred = min_index + window.shape[0]
                    labels.extend(
                        label_col.iloc[last_pred : last_pred + window.shape[0]]
                    )
                    break

                distance_profile = self.multi_dimensional_mass(
                    window, series_to_be_searched
                )
                min_index = distance_profile.argmin()
                labels.extend(label_col.iloc[min_index : min_index + window.shape[0]])

        return np.array(labels)

    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""

        if self._is_trained:
            y_test = test_data[self.data_schema.target].to_numpy()
            test_data = test_data.drop(columns=[self.data_schema.target])
            prediction = self.predict(test_data)
            f1 = f1_score(y_test, prediction, average="weighted")
            return f1

        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the TimeStepClassifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TimeStepClassifier":
        """Load the TimeStepClassifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            TimeStepClassifier: A new instance of the loaded TimeStepClassifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model


def train_predictor_model(
    train_data: np.ndarray,
    data_schema: TimeStepClassificationSchema,
    hyperparameters: dict,
) -> TimeStepClassifier:
    """
    Instantiate and train the TimeStepClassifier model.

    Args:
        train_data (np.ndarray): The train split from training data.
        data_schema (TimeStepClassificationSchema): The schema of the data.
        hyperparameters (dict): Hyperparameters for the TimeStepClassifier.

    Returns:
        'TimeStepClassifier': The TimeStepClassifier model
    """
    model = TimeStepClassifier(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(model: TimeStepClassifier, test_data: np.ndarray) -> np.ndarray:
    """
    Predict the test data using the TimeStepClassifier model

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model.
        test_data (np.ndarray): The test input data for annotation.

    Returns:
        np.ndarray: The annotated data.
    """
    return model.predict(test_data)


def save_predictor_model(model: TimeStepClassifier, predictor_dir_path: str) -> None:
    """
    Save the TimeStepClassifier model to disk.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TimeStepClassifier:
    """
    Load the TimeStepClassifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        TimeStepClassifier: A new instance of the loaded TimeStepClassifier model.
    """
    return TimeStepClassifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: TimeStepClassifier, test_split: np.ndarray
) -> float:
    """
    Evaluate the TimeStepClassifier model and return the f1-score value.

    Args:
        model (TimeStepClassifier): The TimeStepClassifier model.
        test_split (np.ndarray): Test data.

    Returns:
        float: The f1-score value of the TimeStepClassifier model.
    """
    return model.evaluate(test_split)


class InsufficientDataError(Exception):
    """Raised when the data length is less that encode length"""
