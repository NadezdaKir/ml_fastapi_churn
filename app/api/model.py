import logging
from typing import Annotated

from fastapi import APIRouter, Body, Query

from app.core.errors import ChurnServiceError
from app.core.state import model_store
from app.ml.dataset import load_dataset
from app.ml.history import add_history_record, get_history
from app.ml.model import evaluate_model, save_churn_model, train_churn_model
from app.ml.preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES, prepare_and_split
from app.schemas import (
    FeatureVectorChurn,
    TRAIN_ERROR_RESPONSES,
    TRAIN_EXAMPLES,
    TrainingConfigChurn,
)

logger = logging.getLogger("churn_service")

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train", responses=TRAIN_ERROR_RESPONSES)
def model_train(
    config: Annotated[TrainingConfigChurn, Body(openapi_examples=TRAIN_EXAMPLES)],
):
    logger.info(
        "training started: model_type=%s hyperparameters=%s",
        config.model_type,
        config.hyperparameters,
    )
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_and_split(df)

    try:
        pipeline = train_churn_model(
            X_train, y_train, config.model_type, config.hyperparameters
        )
    except (ValueError, TypeError) as error:
        raise ChurnServiceError(
            400,
            "invalid_training_config",
            "invalid model type or hyperparameters",
            {"reason": str(error)},
        )

    metrics = evaluate_model(pipeline, X_test, y_test)
    model_store["model_data"] = save_churn_model(
        pipeline, metrics, config.model_type, config.hyperparameters
    )
    add_history_record(model_store["model_data"])
    logger.info("training finished: %s metrics=%s", config.model_type, metrics)
    return metrics


@router.get("/status")
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


@router.get("/schema")
def model_schema():
    fields = FeatureVectorChurn.model_fields
    return {
        "features": [
            {
                "name": name,
                "type": fields[name].annotation.__name__,
                "kind": "numeric" if name in NUMERIC_FEATURES else "categorical",
            }
            for name in FEATURE_COLUMNS
        ]
    }


@router.get("/metrics")
def model_metrics(
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
    model_type: str | None = None,
):
    history = get_history(model_type)
    return {
        "latest": history[0] if history else None,
        "best_by_f1": max(history, key=lambda r: r["metrics"]["f1"]) if history else None,
        "history": history[:limit],
    }