from hw3.spam_baseline import data


def test_load_from_string():
    csv = "ham,Hello there\nspam,Buy now!"
    df = data.load_from_string(csv)
    assert list(df.columns)[:2] == ["label", "message"]
    X_train, X_test, y_train, y_test = data.prepare_dataset(df, test_size=0.5, random_state=0)
    assert len(X_train) == 1
    assert len(X_test) == 1
