import json
import os

from app.core import config


def load_history(path=None):
    path = path or config.HISTORY_PATH
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def add_history_record(model_data, path=None):
    path = path or config.HISTORY_PATH
    history = load_history(path)
    record = {
        "timestamp": model_data["trained_at"],
        "model_type": model_data["model_type"],
        "hyperparameters": model_data["hyperparameters"],
        "metrics": model_data["metrics"],
    }
    history.append(record)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    return record


def get_history(model_type=None, path=None):
    history = load_history(path)
    if model_type is not None:
        history = [record for record in history if record["model_type"] == model_type]
    return list(reversed(history))