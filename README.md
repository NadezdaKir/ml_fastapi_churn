# Churn Prediction Service

Сервис на FastAPI предсказывает отток клиентов. По поведению и характеристикам клиента модель оценивает, уйдёт ли он из сервиса в следующем месяце, и возвращает класс и вероятности.

Сервис умеет обучать модель на датасете, сохранять её на диск, отдавать предсказания для одного клиента или списка, показывать метрики и историю обучений.

## Датасет

Файл `data/churn_dataset.csv` содержит 2000 строк и 10 столбцов.

| Признак | Тип | Описание |
|---|---|---|
| `monthly_fee` | float | ежемесячная стоимость тарифа |
| `usage_hours` | float | часы использования за последний месяц |
| `support_requests` | int | обращения в техподдержку |
| `account_age_months` | int | возраст аккаунта в месяцах |
| `failed_payments` | int | неудачные платежи |
| `region` | str | europe, asia, america, africa |
| `device_type` | str | mobile, desktop, tablet |
| `payment_method` | str | card, paypal, crypto |
| `autopay_enabled` | int | автосписание, 0 или 1 |
| `churn` | int | целевой признак, 1 клиент ушёл, 0 остался |

Классы несбалансированы, ушедших клиентов около 20%.

## Структура проекта

```
app/
  main.py          сборка приложения
  schemas.py       Pydantic модели запросов и ответов
  api/             эндпоинты по разделам
  ml/              датасет, предобработка, обучение, история
  core/            настройки, ошибки, хранилище модели
data/              датасет
tests/             тесты pytest
Dockerfile
requirements.txt
```

## Запуск локально

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Документация откроется по адресу http://127.0.0.1:8000/docs

## Запуск в Docker

```bash
docker build -t churn-service .
docker run -p 8000:8000 churn-service
```

Модель в образ не входит, после запуска контейнера её нужно обучить через `/model/train`.

## Тесты

```bash
pip install pytest httpx
pytest -v
```

Тесты работают на подвыборке из 200 строк во временной папке и не трогают сохранённую модель.

## Основные эндпоинты

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | состояние сервиса |
| GET | `/dataset/info` | размер датасета и распределение классов |
| POST | `/model/train` | обучение модели |
| GET | `/model/status` | статус текущей модели |
| GET | `/model/metrics` | история обучений и лучшая модель |
| GET | `/model/schema` | список признаков и типов |
| POST | `/predict` | предсказание для клиента или списка |

## Примеры запросов

Обучение логистической регрессии

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {"max_iter": 1000, "class_weight": "balanced"}}'
```

Ответ

```json
{"accuracy": 0.59, "f1": 0.354, "roc_auc": 0.611}
```

Доступные модели `logreg` и `random_forest`.

Предсказание для одного клиента

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 29.99, "usage_hours": 3.5, "support_requests": 5, "account_age_months": 2, "failed_payments": 3, "region": "asia", "device_type": "mobile", "payment_method": "crypto", "autopay_enabled": 0}'
```

Ответ

```json
{"churn": 1, "probability_no_churn": 0.063, "probability_churn": 0.937}
```

Для нескольких клиентов в теле запроса передаётся список таких объектов, в ответ приходит список предсказаний.

## Ошибки

Все ошибки возвращаются в едином формате

```json
{"code": "model_not_trained", "message": "model is not trained", "details": {"hint": "call POST /model/train first"}}
```
