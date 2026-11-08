import logging

DATASET_PATH = "data/churn_dataset.csv"
MODEL_PATH = "models/churn_model.joblib"
HISTORY_PATH = "models/training_history.json"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )