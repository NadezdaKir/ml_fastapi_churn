import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import dataset, health, model, predict
from app.core.config import setup_logging
from app.core.errors import register_error_handlers
from app.core.state import model_store
from app.ml.model import load_churn_model

setup_logging()
logger = logging.getLogger("churn_service")


@asynccontextmanager
async def lifespan(app):
    model_store["model_data"] = load_churn_model()
    if model_store["model_data"] is None:
        logger.info("no saved model found, service starts without a model")
    else:
        logger.info("model loaded: %s", model_store["model_data"].get("model_type"))
    yield


app = FastAPI(title="Churn Prediction Service", lifespan=lifespan)

register_error_handlers(app)

app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(predict.router)