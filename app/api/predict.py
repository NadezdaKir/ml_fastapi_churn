import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Body

from app.core.errors import ChurnServiceError
from app.core.state import model_store
from app.ml.preprocessing import FEATURE_COLUMNS
from app.schemas import (
    FeatureVectorChurn,
    PREDICT_ERROR_RESPONSES,
    PREDICT_EXAMPLES,
    PredictionResponseChurn,
)

logger = logging.getLogger("churn_service")

router = APIRouter(tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    responses=PREDICT_ERROR_RESPONSES,
)
def predict(
    features: Annotated[
        FeatureVectorChurn | list[FeatureVectorChurn],
        Body(openapi_examples=PREDICT_EXAMPLES),
    ],
):
    model_data = model_store["model_data"]
    if model_data is None:
        raise ChurnServiceError(
            503,
            "model_not_trained",
            "model is not trained",
            {"hint": "call POST /model/train first"},
        )

    items = features if isinstance(features, list) else [features]
    if not items:
        raise ChurnServiceError(400, "empty_input", "no clients to predict")

    logger.info("predict called for %d client(s)", len(items))

    try:
        df = pd.DataFrame([item.model_dump() for item in items])[FEATURE_COLUMNS]
        pipeline = model_data["pipeline"]
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)
    except Exception as error:
        raise ChurnServiceError(
            500,
            "prediction_error",
            "failed to make a prediction",
            {"reason": str(error)},
        )

    results = [
        PredictionResponseChurn(
            churn=int(pred),
            probability_no_churn=float(proba[0]),
            probability_churn=float(proba[1]),
        )
        for pred, proba in zip(predictions, probabilities)
    ]
    return results if isinstance(features, list) else results[0]