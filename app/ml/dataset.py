import logging
import os

import pandas as pd
from pandas.errors import EmptyDataError

from app.core import config
from app.core.errors import ChurnServiceError
from app.schemas import DatasetRowChurn

logger = logging.getLogger("churn_service")


def load_dataset():
    path = config.DATASET_PATH
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise ChurnServiceError(
            404, "dataset_not_found", "dataset file not found", {"path": path}
        )
    except EmptyDataError:
        raise ChurnServiceError(400, "dataset_empty", "dataset is empty")

    if df.empty:
        raise ChurnServiceError(400, "dataset_empty", "dataset is empty")

    logger.info("dataset loaded: %d rows from %s", len(df), path)
    return df


def dataset_available():
    return os.path.exists(config.DATASET_PATH)


def to_rows(df):
    return [DatasetRowChurn(**row) for row in df.to_dict(orient="records")]