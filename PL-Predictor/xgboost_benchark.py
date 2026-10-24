from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from prepare_data import (
    get_base_feature_columns,
    get_venue_feature_columns,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "model_data.csv"

TEST_SEASON = "Season2025-2026"


def multiclass_brier(y_true, probabilities):
    actual = np.zeros_like(probabilities)

    actual[
        np.arange(len(y_true)),
        y_true.astype(int),
    ] = 1

    return np.mean(
        np.sum(
            (probabilities - actual) ** 2,
            axis=1,
        )
    )


def evaluate(name, y_true, probabilities):
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    print()
    print(name)
    print("=" * 40)

    print(
        f"Accuracy: "
        f"{accuracy_score(y_true, predictions):.3f}"
    )

    print(
        f"Log loss: "
        f"{log_loss(y_true, probabilities):.3f}"
    )

    print(
        f"Brier: "
        f"{multiclass_brier(y_true, probabilities):.3f}"
    )


df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(
    df["Date"]
)


# Start with the simpler feature set first.
features = get_base_feature_columns()


test = df[
    df["Season"] == TEST_SEASON
].copy()

test_start = test["Date"].min()

train = df[
    df["Date"] < test_start
].copy()


train = train.dropna(
    subset=features + ["Target"]
)

test = test.dropna(
    subset=features + ["Target"]
)


X_train = train[features]
y_train = train["Target"]

X_test = test[features]
y_test = test["Target"].to_numpy()


model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.1,
    reg_lambda=1.0,

    random_state=42,
    eval_metric="mlogloss",
)


model.fit(
    X_train,
    y_train,
)


probabilities = model.predict_proba(
    X_test
)


evaluate(
    "XGBoost",
    y_test,
    probabilities,
)