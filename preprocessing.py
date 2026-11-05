from sklearn.model_selection import train_test_split

TARGET = "churn"

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES = [
    "region",
    "device_type",
    "payment_method",
]


def prepare_data(df):
    df = df.copy()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna(
        df[CATEGORICAL_FEATURES].mode().iloc[0]
    )
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def class_distribution(y):
    return {int(k): int(v) for k, v in y.value_counts().items()}
