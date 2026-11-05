import pandas as pd

from schemas import DatasetRowChurn

DATASET_PATH = "data/churn_dataset.csv"


def load_dataset():
    return pd.read_csv(DATASET_PATH)


def to_rows(df):
    return [DatasetRowChurn(**row) for row in df.to_dict(orient="records")]