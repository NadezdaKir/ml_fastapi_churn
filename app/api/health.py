from fastapi import APIRouter

from app.core.state import model_store
from app.ml.dataset import dataset_available

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"message": "ml churn service is running"}


@router.get("/health")
def health():
    model_ok = model_store["model_data"] is not None
    dataset_ok = dataset_available()
    return {
        "status": "ok" if model_ok and dataset_ok else "degraded",
        "model_available": model_ok,
        "dataset_available": dataset_ok,
    }