import joblib
import numpy as np
import pandas as pd
from typing import Any, Tuple, Dict
from sklearn.pipeline import Pipeline
from preprocessing import custom_transformers as transformers


TRAINING_PIPELINE_FILE_NAME = "training_pipeline.joblib"
INFERENCE_PIPELINE_FILE_NAME = "inference_pipeline.joblib"


def create_preprocess_pipeline(
    data_schema: Any,
    scaler_type: str,
) -> Tuple[Pipeline, Pipeline]:
    """
    Constructs two preprocessing pipeline for time-series data:
        one for training and another inference.

    Args:
        data_schema: The schema of the data, containing information like column names.
        preprocessing_config (dict): Configuration parameters for preprocessing, like scaler bounds.
        scaler (int): Scaler to use for normalizing features.

    Returns:
        Tuple[Pipeline, Pipeline]: Tuple of training and inference pipelines
    """
    # Steps for preprocessing pipeline
    steps = [
        (
            "column_selector",
            transformers.ColumnSelector(
                columns=(
                    [data_schema.id_col, data_schema.time_col, data_schema.target]
                    + data_schema.features
                )
            ),
        ),
        (
            "time_col_caster",
            transformers.TimeColCaster(
                time_col=data_schema.time_col, data_type=data_schema.time_col_dtype
            ),
        ),
        (
            "df_sorter",
            transformers.DataFrameSorter(
                sort_columns=[data_schema.id_col, data_schema.time_col],
                ascending=[True, True],
            ),
        ),
        (
            "scaler",
            transformers.ScalerByFeatureDim(
                columns=data_schema.features,
                scaler_type=scaler_type,
            ),
        ),
        (
            "target_encoder",
            transformers.LabelEncoder(columns=[data_schema.target]),
        ),
    ]
    return Pipeline(steps)


def fit_transform_with_pipeline(
    pipeline: Pipeline, data: pd.DataFrame
) -> Tuple[Pipeline, np.ndarray]:
    """
    Fit the preprocessing pipeline and transform data.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        data (pd.DataFrame): The data as a numpy array

    Returns:
        Pipeline: Fitted preprocessing pipeline.
        np.ndarray: transformed data as a numpy array.
    """
    trained_pipeline = fit_pipeline(pipeline, data)
    transformed_data = transform_data(trained_pipeline, data)
    return trained_pipeline, transformed_data


def fit_pipeline(pipeline: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """
    Train the preprocessing pipeline.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        data (pd.DataFrame): The training data as a pandas DataFrame.

    Returns:
        Pipeline: Fitted preprocessing pipeline.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pd.DataFrame")
    pipeline.fit(data)
    return pipeline


def transform_data(pipeline: Pipeline, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input data using the preprocessing pipeline.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        input_data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
        pd.DataFrame: The transformed data.
    """
    return pipeline.transform(input_data)


def save_pipeline(pipeline: Pipeline, file_path_and_name: str) -> None:
    """Save the fitted pipeline to a pickle file.

    Args:
        pipeline (Pipeline): The fitted pipeline to be saved.
        file_path_and_name (str): The path where the pipeline should be saved.
    """
    joblib.dump(pipeline, file_path_and_name)


def load_pipeline(file_path_and_name: str) -> Pipeline:
    """Load the fitted pipeline from the given path.

    Args:
        file_path_and_name: Path to the saved pipeline.

    Returns:
        Fitted pipeline.
    """
    return joblib.load(file_path_and_name)
