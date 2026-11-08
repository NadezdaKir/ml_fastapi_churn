from app.core import config
from app.schemas import LOYAL_CLIENT, RISKY_CLIENT

TRAIN_CONFIG = {
    "model_type": "logreg",
    "hyperparameters": {"max_iter": 1000, "class_weight": "balanced"},
}


def test_full_scenario(client):
    info = client.get("/dataset/info")
    assert info.status_code == 200
    assert info.json()["n_rows"] == 200

    train = client.post("/model/train", json=TRAIN_CONFIG)
    assert train.status_code == 200
    assert set(train.json()) == {"accuracy", "f1", "roc_auc"}

    status = client.get("/model/status").json()
    assert status["is_trained"] is True
    assert status["model_type"] == "logreg"

    prediction = client.post("/predict", json=RISKY_CLIENT)
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["churn"] in (0, 1)
    assert abs(body["probability_churn"] + body["probability_no_churn"] - 1) < 1e-6


def test_predict_batch(client):
    client.post("/model/train", json=TRAIN_CONFIG)
    response = client.post("/predict", json=[LOYAL_CLIENT, RISKY_CLIENT])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_status_without_model(client):
    assert client.get("/model/status").json()["is_trained"] is False


def test_predict_without_model(client):
    response = client.post("/predict", json=RISKY_CLIENT)
    assert response.status_code == 503
    assert response.json()["code"] == "model_not_trained"


def test_predict_wrong_type(client):
    bad_client = {**RISKY_CLIENT, "monthly_fee": "дорого"}
    response = client.post("/predict", json=bad_client)
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_predict_missing_feature(client):
    bad_client = {k: v for k, v in RISKY_CLIENT.items() if k != "region"}
    response = client.post("/predict", json=bad_client)
    assert response.status_code == 422


def test_predict_extra_feature(client):
    bad_client = {**RISKY_CLIENT, "age": 30}
    response = client.post("/predict", json=bad_client)
    assert response.status_code == 422


def test_predict_empty_list(client):
    client.post("/model/train", json=TRAIN_CONFIG)
    response = client.post("/predict", json=[])
    assert response.status_code == 400
    assert response.json()["code"] == "empty_input"


def test_train_unknown_model(client):
    response = client.post("/model/train", json={"model_type": "xgboost"})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_training_config"


def test_train_empty_dataset(client, sample_df):
    sample_df.head(0).to_csv(config.DATASET_PATH, index=False)
    response = client.post("/model/train", json=TRAIN_CONFIG)
    assert response.status_code == 400
    assert response.json()["code"] == "dataset_empty"