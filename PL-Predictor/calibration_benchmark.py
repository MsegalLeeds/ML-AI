from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import get_base_feature_columns


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "model_data.csv"

VALIDATION_SEASON = "Season2024-2025"
TEST_SEASON = "Season2025-2026"


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
            ),
        ),
    ])


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


def temperature_scale(
    probabilities,
    temperature,
):
    # Avoid log(0).
    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    log_probs = np.log(
        probabilities
    )

    scaled = log_probs / temperature

    scaled -= scaled.max(
        axis=1,
        keepdims=True,
    )

    exp_probs = np.exp(scaled)

    return (
        exp_probs
        / exp_probs.sum(
            axis=1,
            keepdims=True,
        )
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
        f"{log_loss(y, probabilities):.4f}"
    )

    print(
        f"Brier: "
        f"{multiclass_brier(y, probabilities):.4f}"
    )


def main():
    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    features = get_base_feature_columns()

    required = (
        features
        + ["Target"]
    )

    df = df.dropna(
        subset=required
    ).copy()

    # ==========================================
    # VALIDATION — find temperature
    # ==========================================

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

    model = make_model()

    model.fit(
        train_validation[features],
        train_validation["Target"],
    )

    validation_probs = (
        model.predict_proba(
            validation[features]
        )
    )

    y_validation = (
        validation["Target"]
        .astype(int)
        .to_numpy()
    )

    def objective(temperature):
        calibrated = temperature_scale(
            validation_probs,
            temperature,
        )

        return log_loss(
            y_validation,
            calibrated,
        )

    result = minimize_scalar(
        objective,
        bounds=(0.5, 2.0),
        method="bounded",
    )

    best_temperature = result.x

    print()
    print(
        f"Best temperature: "
        f"{best_temperature:.3f}"
    )

    print(
        f"Validation log loss: "
        f"{result.fun:.4f}"
    )

    # ==========================================
    # TEST
    # ==========================================

    test = df[
        df["Season"]
        == TEST_SEASON
    ].copy()

    test_start = test["Date"].min()

    train_test = df[
        df["Date"] < test_start
    ].copy()

    model = make_model()

    model.fit(
        train_test[features],
        train_test["Target"],
    )

    raw_probs = model.predict_proba(
        test[features]
    )

    calibrated_probs = temperature_scale(
        raw_probs,
        best_temperature,
    )

    y_test = (
        test["Target"]
        .astype(int)
        .to_numpy()
    )

    evaluate(
        "Raw Logistic",
        y_test,
        raw_probs,
    )

    evaluate(
        "Calibrated Logistic",
        y_test,
        calibrated_probs,
    )


if __name__ == "__main__":
    main()