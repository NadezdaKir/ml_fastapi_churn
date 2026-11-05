from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pandas.errors import EmptyDataError

from dataset import load_dataset, to_rows
from model import evaluate_model, load_churn_model, save_churn_model, train_churn_model
from preprocessing import class_distribution, prepare_data, split_data
from schemas import (
    FeatureVectorChurn,
    PREDICT_EXAMPLES,
    PredictionResponseChurn,
    TRAIN_EXAMPLES,
    TrainingConfigChurn,
)

model_store = {"model_data": None}


@asynccontextmanager
async def lifespan(app):
    model_store["model_data"] = load_churn_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
)
def predict(
    features: Annotated[
        FeatureVectorChurn | list[FeatureVectorChurn],
        Body(openapi_examples=PREDICT_EXAMPLES),
    ],
):
    model_data = model_store["model_data"]
    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="model is not trained, call POST /model/train first",
        )

    items = features if isinstance(features, list) else [features]
    df = pd.DataFrame([item.model_dump() for item in items])

    pipeline = model_data["pipeline"]
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)

    results = [
        PredictionResponseChurn(
            churn=int(pred),
            probability_no_churn=float(proba[0]),
            probability_churn=float(proba[1]),
        )
        for pred, proba in zip(predictions, probabilities)
    ]
    return results if isinstance(features, list) else results[0]


@app.get("/dataset/preview")
def dataset_preview(n: int = 5):
    df = load_dataset()
    return to_rows(df.head(n))


@app.get("/dataset/info")
def dataset_info():
    df = load_dataset()
    churn_counts = df["churn"].value_counts()
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "features": [col for col in df.columns if col != "churn"],
        "churn_distribution": {int(k): int(v) for k, v in churn_counts.items()},
    }


@app.get("/dataset/split-info")
def dataset_split_info():
    df = load_dataset()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_distribution": class_distribution(y_train),
        "test_churn_distribution": class_distribution(y_test),
    }


@app.post("/model/train")
def model_train(
    config: Annotated[TrainingConfigChurn, Body(openapi_examples=TRAIN_EXAMPLES)],
):
    try:
        df = load_dataset()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="dataset not found")
    except EmptyDataError:
        raise HTTPException(status_code=400, detail="dataset is empty")

    if df.empty:
        raise HTTPException(status_code=400, detail="dataset is empty")

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    try:
        pipeline = train_churn_model(
            X_train, y_train, config.model_type, config.hyperparameters
        )
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    metrics = evaluate_model(pipeline, X_test, y_test)
    model_store["model_data"] = save_churn_model(
        pipeline, metrics, config.model_type, config.hyperparameters
    )
    return metrics


@app.get("/model/status")
def model_status():
    model_data = model_store["model_data"]
    if model_data is None:
        return {
            "is_trained": False,
            "trained_at": None,
            "model_type": None,
            "hyperparameters": None,
            "metrics": None,
        }
    return {
        "is_trained": True,
        "trained_at": model_data["trained_at"],
        "model_type": model_data.get("model_type"),
        "hyperparameters": model_data.get("hyperparameters"),
        "metrics": model_data["metrics"],
    }


@app.get("/model/status")
def model_status():
    model_data = model_store["model_data"]
    if model_data is None:
        return {"is_trained": False, "trained_at": None, "metrics": None}
    return {
        "is_trained": True,
        "trained_at": model_data["trained_at"],
        "metrics": model_data["metrics"],
    }