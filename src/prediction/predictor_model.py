import os
import warnings
import joblib
import stumpy
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from schema.data_schema import TimeStepClassificationSchema
from tqdm import tqdm


from logger import get_logger


logger = get_logger(task_name=__file__)

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"


class TimeStepClassifier:
    """Timeseries Step classifier.

    This class provides a consistent interface that can be used with other
    TimeStepClassifier models.
    """

    MODEL_NAME = "Distance_Profile_Timeseries_classifier"

    def __init__(
        self,
        data_schema: TimeStepClassificationSchema,
        window_length_factor: float = 4.0,
        num_neighbours: int = 3,
        **kwargs,
    ):
        """
        Construct a new Distance Profile TimeStepClassifier.

        Args:
            data_schema (TimeStepClassificationSchema): The schema of the data.
            window_length_factor (float): Factor used to adjust the base window length
                                         derived from the logarithm of the minimum row
                                         count per group.
            num_neighbors (int): Number of neighbors in similarity matches.
        """
        self.data_schema = data_schema
        self.window_length_factor = window_length_factor
        self.num_neighbours = num_neighbours
        self._is_trained = False
        self.train_data = None
        self.window_length = None
        self.stride = None
        self._verify_params()

    def _verify_params(self) -> None:
        assert (
            self.window_length_factor > 0
        ), "Window length factor must be greater than 0."
        assert self.num_neighbours > 0, "Number of neighbors must be greater than 0."

    def create_windows_for_prediction(self, data: np.ndarray) -> np.ndarray:
        """
        Create windows for prediction.

        Args:
            data (np.ndarray): The data to create windows for.
                               Shape is (L, D)
                                where L is the number of rows and
                                D is the number of dimensions.

        Returns:
            np.ndarray: Windowed data of shape [L-window_length+1, window_length, D]
        """
        start_idx_list = []
        windows_list = []
        for i in range(0, len(data), self.stride):
            start_idx = i
            if start_idx + self.window_length > len(data) - 1:
                # last window runs out of space, so slide it in to fit
                start_idx = len(data) - self.window_length
            start_idx_list.append(start_idx)
            windows_list.append(data[start_idx : start_idx + self.window_length])
        return np.stack(windows_list, axis=0), start_idx_list

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
        """
        Fits the model using the provided training data.

        This method processes the training data by sorting it based on the
        specified identifier and time columns. It then calculates the minimum
        number of rows (`min_row_count`) for any unique identifier in the data, 
        computes the base-2  logarithm of this minimum count, and multiplies it 
        by a predefined factor (`window_length_factor`) to determine the window
        length used in later computations. Additionally, it sets the stride length
        to one-third of the window length and marks the model as trained.

        `min_row_count` is calculated across all samples. It represents the minimum
        series length in the training data. 

        Args:
            train_data (pd.DataFrame): The training data containing the columns
                specified by `self.data_schema.id_col` and `self.data_schema.time_col`.

        Attributes:
            train_data (pd.DataFrame): The sorted training data stored for use during
                                       inference.
            window_length (int): The length of the window used for computing distance
                                 profiles,derived from the logarithm of the minimum
                                 row count per group.
            stride (int): The stride length for moving the window across the data,
                set to one-third of the window length.
            _is_trained (bool): A flag indicating whether the model has been trained.

        Examples:
        - For a `min_row_count` of 8, a `window_length_factor` of 2 results in
          a `window_length` of 6 (calculated as `int(log2(8) * 2)`).
        - For a `min_row_count` of 32, a `window_length_factor` of 3 results in
          a `window_length` of 15 (calculated as `int(log2(32) * 3)`).
        - For a `min_row_count` of 100, a `window_length_factor` of 1.5 results in
          a `window_length` of 9 (calculated as `int(log2(100) * 1.5)`).
        
        Raises:
            ValueError: If the training data is missing required columns specified
                by `self.data_schema`.
        """
        self.train_data = train_data.sort_values(
            by=[self.data_schema.id_col, self.data_schema.time_col]
        )
        grouped = train_data.groupby(self.data_schema.id_col).size()
        min_row_count = grouped.min()
        log_min_count = np.log2(min_row_count)
        self.window_length = int(log_min_count * self.window_length_factor)
        self.stride = self.window_length // 3
        logger.info(f"Calculated window length = {self.window_length}")
        logger.info(f"Calculated stride = {self.stride}")
        self._is_trained = True

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Run predictions on test data

        Args:
            test_data (pd.DataFrame): Test dataframe
                Must contain id_col, time_col and feature columns
                as specified in self.data_schema

        Returns:
            np.ndarray: Array of predictions of shape [N, K] where
                        N is number of rows in test_data and K
                        is the number of target classes
        """
        window_length = self.window_length
        series_to_be_searched = self.train_data  # training data
        label_col = series_to_be_searched[self.data_schema.target]
        series_to_be_searched = series_to_be_searched.drop(
            columns=[
                self.data_schema.id_col,
                self.data_schema.time_col,
                self.data_schema.target,
            ]
        ).to_numpy()

        grouped = test_data.groupby(self.data_schema.id_col)

        id_cols = [self.data_schema.id_col, self.data_schema.time_col]
        encoded_target_cols = [
            int(i) for i in range(len(self.data_schema.target_classes))
        ]

        all_preds = []
        for id_, group in tqdm(grouped, desc="Generating predictions"):
            # iterate over samples in test data
            logger.info(f"Running predictions for id = {id_}")

            group_arr = group.drop(
                columns=[self.data_schema.id_col, self.data_schema.time_col]
            ).to_numpy()

            windowed_group, start_idx_list = self.create_windows_for_prediction(
                group_arr
            )

            for i, window in enumerate(windowed_group):
                distance_profile = self.multi_dimensional_mass(
                    window, series_to_be_searched
                )

                # start and end indices of the window in the grouped data
                start_idx = start_idx_list[i]
                end_idx = start_idx + window_length

                # Get the indices of the k smallest values in distance profile
                indices_of_k_smallest = np.argsort(distance_profile)[
                    : self.num_neighbours
                ]
                for _, idx in enumerate(indices_of_k_smallest):
                    # Iterate over k neighbors

                    # Create one-hot encoded predictions; first initialize to zeros
                    pred_probs = np.zeros(
                        (window_length, len(self.data_schema.target_classes))
                    )
                    # Get predicted target labels given this neighbor
                    window_labels = label_col.iloc[idx : idx + window_length].values
                    window_labels = window_labels.astype(np.int16)
                    pred_probs[np.arange(pred_probs.shape[0]), window_labels] = 1.0
                    # concatenate one-hot encoded predictions with id and time columns
                    pred_probs = np.concat(
                        [group.iloc[start_idx:end_idx][id_cols].values, pred_probs],
                        axis=1,
                    )
                    all_preds.append(pred_probs)

        # Create overall predictions dataframe containing all test samples
        all_preds_df = pd.DataFrame(np.concat(all_preds, axis=0))
        all_preds_df.columns = id_cols + encoded_target_cols

        # Average by id and time columns since the same time idx can be repeated over
        # many overlapping windows
        averaged_preds = (
            all_preds_df.groupby(id_cols)[encoded_target_cols].mean().reset_index()
        )
        return averaged_preds[encoded_target_cols].values

    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""
        if self._is_trained:
            y_test = test_data[self.data_schema.target].to_numpy()
            test_data = test_data.drop(columns=[self.data_schema.target])
            predictions = self.predict(test_data)
            predictions = np.argmax(predictions, axis=1)
            f1 = f1_score(y_test, predictions, average="weighted")
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
