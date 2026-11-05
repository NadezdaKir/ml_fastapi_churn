from pydantic import BaseModel


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    churn: int
    probability_no_churn: float
    probability_churn: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "churn": 1,
                    "probability_no_churn": 0.28,
                    "probability_churn": 0.72,
                }
            ]
        }
    }


LOYAL_CLIENT = {
    "monthly_fee": 19.99,
    "usage_hours": 45.0,
    "support_requests": 0,
    "account_age_months": 36,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1,
}

RISKY_CLIENT = {
    "monthly_fee": 29.99,
    "usage_hours": 3.5,
    "support_requests": 5,
    "account_age_months": 2,
    "failed_payments": 3,
    "region": "asia",
    "device_type": "mobile",
    "payment_method": "crypto",
    "autopay_enabled": 0,
}

PREDICT_EXAMPLES = {
    "single": {
        "summary": "Один клиент",
        "value": RISKY_CLIENT,
    },
    "batch": {
        "summary": "Несколько клиентов",
        "value": [LOYAL_CLIENT, RISKY_CLIENT],
    },
}

class TrainingConfigChurn(BaseModel):
    model_type: str = "logreg"
    hyperparameters: dict = {}


TRAIN_EXAMPLES = {
    "logreg": {
        "summary": "Логистическая регрессия",
        "value": {
            "model_type": "logreg",
            "hyperparameters": {"max_iter": 1000, "class_weight": "balanced"},
        },
    },
    "random_forest": {
        "summary": "Случайный лес",
        "value": {
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 6,
                "class_weight": "balanced",
                "random_state": 42,
            },
        },
    },
}