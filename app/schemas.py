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
    model_config = {"extra": "forbid"}


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

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict | list | None = None


def error_example(summary, code, message, details=None):
    return {
        "summary": summary,
        "value": {"code": code, "message": message, "details": details},
    }


def error_response_doc(description, examples):
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"examples": examples}},
    }


VALIDATION_ERROR_EXAMPLE = error_example(
    "Неверные данные в запросе",
    "validation_error",
    "request data does not match the expected schema",
    [{"type": "missing", "loc": ["body", "monthly_fee"], "msg": "Field required"}],
)

TRAIN_ERROR_RESPONSES = {
    400: error_response_doc(
        "Пустой датасет или неверная конфигурация обучения",
        {
            "dataset_empty": error_example(
                "Пустой датасет", "dataset_empty", "dataset is empty"
            ),
            "invalid_training_config": error_example(
                "Неизвестная модель",
                "invalid_training_config",
                "invalid model type or hyperparameters",
                {"reason": "unknown model_type: xgboost, available: ['logreg', 'random_forest']"},
            ),
        },
    ),
    404: error_response_doc(
        "Файл датасета не найден",
        {
            "dataset_not_found": error_example(
                "Нет файла",
                "dataset_not_found",
                "dataset file not found",
                {"path": "data/churn_dataset.csv"},
            ),
        },
    ),
    422: error_response_doc(
        "Тело запроса не соответствует схеме",
        {"validation_error": VALIDATION_ERROR_EXAMPLE},
    ),
}

PREDICT_ERROR_RESPONSES = {
    400: error_response_doc(
        "Пустой список клиентов",
        {
            "empty_input": error_example(
                "Пустой список", "empty_input", "no clients to predict"
            ),
        },
    ),
    422: error_response_doc(
        "Неверное количество признаков или неверные типы",
        {
            "missing_field": VALIDATION_ERROR_EXAMPLE,
            "extra_field": error_example(
                "Лишний признак",
                "validation_error",
                "request data does not match the expected schema",
                [{"type": "extra_forbidden", "loc": ["body", "age"], "msg": "Extra inputs are not permitted"}],
            ),
            "wrong_type": error_example(
                "Неверный тип",
                "validation_error",
                "request data does not match the expected schema",
                [{"type": "float_parsing", "loc": ["body", "monthly_fee"], "msg": "Input should be a valid number"}],
            ),
        },
    ),
    500: error_response_doc(
        "Ошибка при предсказании",
        {
            "prediction_error": error_example(
                "Модель не смогла предсказать",
                "prediction_error",
                "failed to make a prediction",
                {"reason": "..."},
            ),
        },
    ),
    503: error_response_doc(
        "Модель ещё не обучена",
        {
            "model_not_trained": error_example(
                "Нет модели",
                "model_not_trained",
                "model is not trained",
                {"hint": "call POST /model/train first"},
            ),
        },
    ),
}