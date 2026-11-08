from fastapi import APIRouter

from app.ml.dataset import load_dataset, to_rows
from app.ml.preprocessing import TARGET, class_distribution, prepare_and_split

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def dataset_preview(n: int = 5):
    df = load_dataset()
    return to_rows(df.head(n))


@router.get("/info")
def dataset_info():
    df = load_dataset()
    churn_counts = df[TARGET].value_counts()
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "features": [col for col in df.columns if col != TARGET],
        "churn_distribution": {int(k): int(v) for k, v in churn_counts.items()},
    }


@router.get("/split-info")
def dataset_split_info():
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_and_split(df)
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_distribution": class_distribution(y_train),
        "test_churn_distribution": class_distribution(y_test),
    }