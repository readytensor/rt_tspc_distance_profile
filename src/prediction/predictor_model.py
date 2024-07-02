import os
import warnings
import sys
import joblib
import stumpy
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from multiprocessing import cpu_count
from sklearn.metrics import f1_score
from schema.data_schema import TSAnnotationSchema
from typing import Tuple
from tqdm import tqdm

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
print(f"Using n_jobs = {n_jobs}")


class TSAnnotator:
    """Random Forest Timeseries Annotator.

    This class provides a consistent interface that can be used with other
    TSAnnotator models.
    """

    MODEL_NAME = "Random_Forest_Timeseries_Annotator"

    def __init__(
        self,
        data_schema: TSAnnotationSchema,
        encode_len: int,
        **kwargs,
    ):
        """
        Construct a new Random Forest TSAnnotator.

        Args:
            data_schema (TSAnnotationSchema): The schema of the data.
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
            for window in windowed_group:
                if window.shape[0] < self.encode_len:

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
        """Save the Random Forest TSAnnotator to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TSAnnotator":
        """Load the Random Forest TSAnnotator from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            TSAnnotator: A new instance of the loaded Random Forest TSAnnotator.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model


def train_predictor_model(
    train_data: np.ndarray,
    data_schema: TSAnnotationSchema,
    hyperparameters: dict,
) -> TSAnnotator:
    """
    Instantiate and train the TSAnnotator model.

    Args:
        train_data (np.ndarray): The train split from training data.
        hyperparameters (dict): Hyperparameters for the TSAnnotator.

    Returns:
        'TSAnnotator': The TSAnnotator model
    """
    model = TSAnnotator(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(model: TSAnnotator, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (TSAnnotator): The TSAnnotator model.
        test_data (np.ndarray): The test input data for annotation.

    Returns:
        np.ndarray: The annotated data.
    """
    return model.predict(test_data)


def save_predictor_model(model: TSAnnotator, predictor_dir_path: str) -> None:
    """
    Save the TSAnnotator model to disk.

    Args:
        model (TSAnnotator): The TSAnnotator model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TSAnnotator:
    """
    Load the TSAnnotator model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        TSAnnotator: A new instance of the loaded TSAnnotator model.
    """
    return TSAnnotator.load(predictor_dir_path)


def evaluate_predictor_model(model: TSAnnotator, test_split: np.ndarray) -> float:
    """
    Evaluate the TSAnnotator model and return the r-squared value.

    Args:
        model (TSAnnotator): The TSAnnotator model.
        test_split (np.ndarray): Test data.

    Returns:
        float: The r-squared value of the TSAnnotator model.
    """
    return model.evaluate(test_split)
