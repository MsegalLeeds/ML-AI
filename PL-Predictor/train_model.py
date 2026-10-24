from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

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
MODEL_FILE = BASE_DIR / "premier_league_model.joblib"

VALIDATION_SEASON = "Season2024-2025"
TEST_SEASON = "Season2025-2026"


def make_logistic():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
            ),
        ),
    ])


def make_poisson():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "poisson",
            PoissonRegressor(
                alpha=1.0,
                max_iter=5000,
            ),
        ),
    ])


def temperature_scale(
    probabilities,
    temperature,
):
    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    logits = np.log(probabilities)

    logits /= temperature

    logits -= logits.max(
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(logits)

    return (
        exp_logits
        / exp_logits.sum(
            axis=1,
            keepdims=True,
        )
    )


def multiclass_brier(
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


def main():
    df = pd.read_csv(DATA_FILE)

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
    # Find calibration temperature on 2024/25
    # ========================================

    validation = df[
        df["Season"]
        == VALIDATION_SEASON
    ].copy()

    validation_start = (
        validation["Date"].min()
    )

    train_validation = df[
        df["Date"] < validation_start
    ].copy()

    calibration_model = make_logistic()

    calibration_model.fit(
        train_validation[features],
        train_validation["Target"],
    )

    validation_probs = (
        calibration_model.predict_proba(
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

    temperature = float(result.x)

    print()
    print(
        f"Temperature: {temperature:.3f}"
    )

    print(
        f"Validation log loss: "
        f"{result.fun:.4f}"
    )

    # ========================================
    # Honest 2025/26 test
    # ========================================

    test = df[
        df["Season"]
        == TEST_SEASON
    ].copy()

    test_start = test["Date"].min()

    train_test = df[
        df["Date"] < test_start
    ].copy()

    test_model = make_logistic()

    test_model.fit(
        train_test[features],
        train_test["Target"],
    )

    raw_probs = test_model.predict_proba(
        test[features]
    )

    calibrated_probs = temperature_scale(
        raw_probs,
        temperature,
    )

    y_test = (
        test["Target"]
        .astype(int)
        .to_numpy()
    )

    predictions = np.argmax(
        calibrated_probs,
        axis=1,
    )

    print()
    print("Calibrated Logistic Test")
    print("=" * 40)

    print(
        f"Accuracy: "
        f"{accuracy_score(
            y_test,
            predictions
        ):.3f}"
    )

    print(
        f"Log loss: "
        f"{log_loss(
            y_test,
            calibrated_probs
        ):.4f}"
    )

    print(
        f"Brier: "
        f"{multiclass_brier(
            y_test,
            calibrated_probs
        ):.4f}"
    )

    # ========================================
    # Production models
    # ========================================

    logistic_model = make_logistic()
    home_goal_model = make_poisson()
    away_goal_model = make_poisson()

    logistic_model.fit(
        df[features],
        df["Target"],
    )

    home_goal_model.fit(
        df[features],
        df["FTHG"],
    )

    away_goal_model.fit(
        df[features],
        df["FTAG"],
    )

    artifact = {
        "model_name":
            "calibrated_logistic_elo",

        "logistic_model":
            logistic_model,

        "home_goal_model":
            home_goal_model,

        "away_goal_model":
            away_goal_model,

        "temperature":
            temperature,

        "features":
            features,

        "trained_through":
            df["Date"].max().isoformat(),
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
    )

    print()
    print(
        f"Saved production model:"
    )
    print(MODEL_FILE)


if __name__ == "__main__":
    main()