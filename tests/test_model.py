import pytest

from app.ml.model import evaluate_model, load_churn_model, save_churn_model, train_churn_model
from app.ml.preprocessing import prepare_data, split_data

CONFIGS = [
    ("logreg", {"max_iter": 1000, "class_weight": "balanced"}),
    ("random_forest", {"n_estimators": 20, "random_state": 42}),
]


@pytest.fixture
def split(sample_df):
    X, y = prepare_data(sample_df)
    return split_data(X, y)


@pytest.mark.parametrize("model_type, hyperparameters", CONFIGS)
def test_train_returns_fitted_pipeline(split, model_type, hyperparameters):
    X_train, X_test, y_train, y_test = split
    pipeline = train_churn_model(X_train, y_train, model_type, hyperparameters)
    probabilities = pipeline.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)


def test_evaluate_returns_metrics_in_range(split):
    X_train, X_test, y_train, y_test = split
    pipeline = train_churn_model(X_train, y_train, *CONFIGS[0])
    metrics = evaluate_model(pipeline, X_test, y_test)
    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())


def test_unknown_model_type_raises(split):
    X_train, X_test, y_train, y_test = split
    with pytest.raises(ValueError):
        train_churn_model(X_train, y_train, "xgboost", {})


def test_save_and_load_model(split, tmp_path):
    X_train, X_test, y_train, y_test = split
    pipeline = train_churn_model(X_train, y_train, *CONFIGS[0])
    metrics = evaluate_model(pipeline, X_test, y_test)
    path = str(tmp_path / "model.joblib")

    save_churn_model(pipeline, metrics, *CONFIGS[0], path=path)
    loaded = load_churn_model(path=path)

    assert loaded["model_type"] == "logreg"
    assert loaded["metrics"] == metrics
    assert (loaded["pipeline"].predict(X_test) == pipeline.predict(X_test)).all()