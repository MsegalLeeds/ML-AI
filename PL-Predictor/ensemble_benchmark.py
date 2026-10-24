from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson

from sklearn.linear_model import (
    LogisticRegression,
    PoissonRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import get_base_feature_columns


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "model_data.csv"

VALIDATION_SEASON = "Season2024-2025"
TEST_SEASON = "Season2025-2026"

MAX_GOALS = 10


def make_logistic():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
            ),
        ),
    ])


def make_poisson():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            PoissonRegressor(
                alpha=1.0,
                max_iter=5000,
            ),
        ),
    ])


def brier_score(
    y_true,
    probabilities,
):
    actual = np.zeros_like(
        probabilities
    )

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


def poisson_result_probabilities(
    home_xg,
    away_xg,
):
    goals = np.arange(
        MAX_GOALS + 1
    )

    home = poisson.pmf(
        goals,
        home_xg,
    )

    away = poisson.pmf(
        goals,
        away_xg,
    )

    matrix = np.outer(
        home,
        away,
    )

    matrix /= matrix.sum()

    home_win = np.tril(
        matrix,
        -1,
    ).sum()

    draw = np.trace(
        matrix
    )

    away_win = np.triu(
        matrix,
        1,
    ).sum()

    return np.array([
        home_win,
        draw,
        away_win,
    ])


def train_models(
    train,
    features,
):
    logistic = make_logistic()

    logistic.fit(
        train[features],
        train["Target"],
    )

    home_model = make_poisson()
    away_model = make_poisson()

    home_model.fit(
        train[features],
        train["FTHG"],
    )

    away_model.fit(
        train[features],
        train["FTAG"],
    )

    return (
        logistic,
        home_model,
        away_model,
    )


def get_probabilities(
    dataset,
    features,
    logistic,
    home_model,
    away_model,
):
    logistic_probs = (
        logistic.predict_proba(
            dataset[features]
        )
    )

    home_xg = np.clip(
        home_model.predict(
            dataset[features]
        ),
        0.05,
        6.0,
    )

    away_xg = np.clip(
        away_model.predict(
            dataset[features]
        ),
        0.05,
        6.0,
    )

    poisson_probs = np.vstack([
        poisson_result_probabilities(
            h,
            a,
        )
        for h, a
        in zip(home_xg, away_xg)
    ])

    return (
        logistic_probs,
        poisson_probs,
    )


def evaluate(
    name,
    y,
    probabilities,
):
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    print()
    print(name)
    print("=" * 40)

    print(
        f"Accuracy: "
        f"{accuracy_score(y, predictions):.3f}"
    )

    print(
        f"Log loss: "
        f"{log_loss(y, probabilities):.3f}"
    )

    print(
        f"Brier: "
        f"{brier_score(y, probabilities):.3f}"
    )


def main():
    df = pd.read_csv(
        DATA_FILE
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    features = (
        get_base_feature_columns()
    )

    required = (
        features
        + [
            "Target",
            "FTHG",
            "FTAG",
        ]
    )

    df = df.dropna(
        subset=required
    ).copy()


    # ========================================
    # VALIDATION
    # Choose ensemble weight using 2024/25
    # ========================================

    validation = df[
        df["Season"]
        == VALIDATION_SEASON
    ].copy()

    validation_start = (
        validation["Date"].min()
    )

    train_validation = df[
        df["Date"]
        < validation_start
    ].copy()

    (
        logistic,
        home_model,
        away_model,
    ) = train_models(
        train_validation,
        features,
    )

    (
        logistic_probs,
        poisson_probs,
    ) = get_probabilities(
        validation,
        features,
        logistic,
        home_model,
        away_model,
    )

    y_validation = (
        validation["Target"]
        .astype(int)
        .to_numpy()
    )


    # ========================================
    # Search for best ensemble weight
    # ========================================

    best_weight = None
    best_loss = float("inf")

    print()
    print("Validation ensemble search")
    print("=" * 40)

    for weight in np.arange(
        0.0,
        1.01,
        0.05,
    ):
        combined = (
            weight
            * logistic_probs
            +
            (1 - weight)
            * poisson_probs
        )

        loss = log_loss(
            y_validation,
            combined,
        )

        print(
            f"Logistic weight "
            f"{weight:.2f}: "
            f"{loss:.4f}"
        )

        if loss < best_loss:
            best_loss = loss
            best_weight = weight


    print()
    print(
        f"Best Logistic weight: "
        f"{best_weight:.2f}"
    )

    print(
        f"Best Poisson weight: "
        f"{1 - best_weight:.2f}"
    )

    print(
        f"Validation log loss: "
        f"{best_loss:.4f}"
    )


    # ========================================
    # FINAL TEST
    # ========================================

    test = df[
        df["Season"]
        == TEST_SEASON
    ].copy()

    test_start = (
        test["Date"].min()
    )

    train_test = df[
        df["Date"] < test_start
    ].copy()

    (
        logistic,
        home_model,
        away_model,
    ) = train_models(
        train_test,
        features,
    )

    (
        logistic_probs,
        poisson_probs,
    ) = get_probabilities(
        test,
        features,
        logistic,
        home_model,
        away_model,
    )

    ensemble_probs = (
        best_weight
        * logistic_probs
        +
        (1 - best_weight)
        * poisson_probs
    )

    y_test = (
        test["Target"]
        .astype(int)
        .to_numpy()
    )


    # ========================================
    # RESULTS
    # ========================================

    evaluate(
        "Logistic Regression",
        y_test,
        logistic_probs,
    )

    evaluate(
        "Poisson",
        y_test,
        poisson_probs,
    )

    evaluate(
        "Ensemble",
        y_test,
        ensemble_probs,
    )


if __name__ == "__main__":
    main()