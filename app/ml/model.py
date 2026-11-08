import os
from datetime import datetime

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core import config
from app.ml.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def create_classifier(model_type, hyperparameters):
    if model_type not in MODELS:
        raise ValueError(
            f"unknown model_type: {model_type}, available: {list(MODELS)}"
        )
    return MODELS[model_type](**hyperparameters)


def build_pipeline(model_type, hyperparameters):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", create_classifier(model_type, hyperparameters)),
        ]
    )


def train_churn_model(X_train, y_train, model_type="logreg", hyperparameters=None):
    pipeline = build_pipeline(model_type, hyperparameters or {})
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def save_churn_model(pipeline, metrics, model_type, hyperparameters, path=None):
    path = path or config.MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_data = {
        "pipeline": pipeline,
        "metrics": metrics,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(model_data, path)
    return model_data


def load_churn_model(path=None):
    path = path or config.MODEL_PATH
    if not os.path.exists(path):
        return None
    return joblib.load(path)