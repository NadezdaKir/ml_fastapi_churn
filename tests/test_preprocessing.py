from app.ml.preprocessing import FEATURE_COLUMNS, TARGET, prepare_data, split_data


def test_prepare_data_separates_target(sample_df):
    X, y = prepare_data(sample_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert TARGET not in X.columns
    assert len(X) == len(y) == len(sample_df)


def test_prepare_data_fills_missing_values(sample_df):
    df = sample_df.copy()
    df.loc[0, "monthly_fee"] = None
    df.loc[1, "region"] = None
    X, _ = prepare_data(df)
    assert X.isna().sum().sum() == 0


def test_split_sizes_and_proportions(sample_df):
    X, y = prepare_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_test) == 40
    assert len(X_train) + len(X_test) == len(X)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_split_is_reproducible(sample_df):
    X, y = prepare_data(sample_df)
    first_test = split_data(X, y)[1]
    second_test = split_data(X, y)[1]
    assert first_test.index.equals(second_test.index)