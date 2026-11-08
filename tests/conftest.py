import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import app


@pytest.fixture
def sample_df():
    df = pd.read_csv("data/churn_dataset.csv")
    return df.sample(n=200, random_state=42).reset_index(drop=True)


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch, sample_df):
    dataset_path = tmp_path / "churn_sample.csv"
    sample_df.to_csv(dataset_path, index=False)
    monkeypatch.setattr(config, "DATASET_PATH", str(dataset_path))
    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "models" / "churn_model.joblib"))
    monkeypatch.setattr(config, "HISTORY_PATH", str(tmp_path / "models" / "training_history.json"))
    return tmp_path


@pytest.fixture
def client(isolated_paths):
    with TestClient(app) as test_client:
        yield test_client