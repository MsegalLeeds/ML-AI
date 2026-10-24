from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import get_base_feature_columns


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "model_data.csv"

TEST_SEASON = "Season2025-2026"

# Scorelines 0-10 goals are effectively enough for EPL games.
MAX_GOALS = 10


def make_poisson_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "poisson",
            PoissonRegressor(
                alpha=1.0,
                max_iter=5000,
            ),
        ),
    ])


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
            (
                probabilities
                - actual
            ) ** 2,
            axis=1,
        )
    )


def expected_goals_to_probabilities(
    home_xg,
    away_xg,
    max_goals=MAX_GOALS,
):
    """
    Convert expected goals into:
        P(Home win)
        P(Draw)
        P(Away win)
    """

    goals = np.arange(
        0,
        max_goals + 1,
    )

    home_probs = poisson.pmf(
        goals,
        home_xg,
    )

    away_probs = poisson.pmf(
        goals,
        away_xg,
    )

    # Matrix:
    #
    # rows    = home goals
    # columns = away goals
    #
    score_matrix = np.outer(
        home_probs,
        away_probs,
    )

    # Small amount of probability may exist
    # beyond MAX_GOALS, so normalise.
    score_matrix /= score_matrix.sum()

    home_win = np.tril(
        score_matrix,
        k=-1,
    ).sum()

    draw = np.trace(
        score_matrix
    )

    away_win = np.triu(
        score_matrix,
        k=1,
    ).sum()

    return np.array([
        home_win,
        draw,
        away_win,
    ])


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

    test = df[
        df["Season"]
        == TEST_SEASON
    ].copy()

    if test.empty:
        raise ValueError(
            f"No rows found for "
            f"{TEST_SEASON}"
        )

    test_start = test[
        "Date"
    ].min()

    train = df[
        df["Date"] < test_start
    ].copy()

    required = (
        features
        + [
            "Target",
            "FTHG",
            "FTAG",
        ]
    )

    train = train.dropna(
        subset=required
    ).copy()

    test = test.dropna(
        subset=required
    ).copy()

    print(
        f"Training matches: "
        f"{len(train)}"
    )

    print(
        f"Test matches: "
        f"{len(test)}"
    )

    # -----------------------------
    # Train TWO Poisson models
    # -----------------------------

    home_goal_model = (
        make_poisson_model()
    )

    away_goal_model = (
        make_poisson_model()
    )

    home_goal_model.fit(
        train[features],
        train["FTHG"],
    )

    away_goal_model.fit(
        train[features],
        train["FTAG"],
    )

    # -----------------------------
    # Predict expected goals
    # -----------------------------

    expected_home_goals = (
        home_goal_model.predict(
            test[features]
        )
    )

    expected_away_goals = (
        away_goal_model.predict(
            test[features]
        )
    )

    # Avoid pathological values.
    expected_home_goals = np.clip(
        expected_home_goals,
        0.05,
        6.0,
    )

    expected_away_goals = np.clip(
        expected_away_goals,
        0.05,
        6.0,
    )

    # -----------------------------
    # Convert goals → W/D/L
    # -----------------------------

    probabilities = np.vstack([
        expected_goals_to_probabilities(
            home_xg,
            away_xg,
        )
        for home_xg, away_xg
        in zip(
            expected_home_goals,
            expected_away_goals,
        )
    ])

    y_test = (
        test["Target"]
        .astype(int)
        .to_numpy()
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    # -----------------------------
    # Evaluation
    # -----------------------------

    print()
    print("Poisson Model")
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
            probabilities
        ):.3f}"
    )

    print(
        f"Brier: "
        f"{multiclass_brier(
            y_test,
            probabilities
        ):.3f}"
    )

    # -----------------------------
    # Show some predictions
    # -----------------------------

    results = test[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]
    ].copy()

    results["ExpHomeGoals"] = (
        expected_home_goals
    )

    results["ExpAwayGoals"] = (
        expected_away_goals
    )

    results["HomeProb"] = (
        probabilities[:, 0]
    )

    results["DrawProb"] = (
        probabilities[:, 1]
    )

    results["AwayProb"] = (
        probabilities[:, 2]
    )

    print()
    print(
        results.tail(20).to_string(
            index=False,
            formatters={
                "ExpHomeGoals":
                    "{:.2f}".format,
                "ExpAwayGoals":
                    "{:.2f}".format,
                "HomeProb":
                    "{:.1%}".format,
                "DrawProb":
                    "{:.1%}".format,
                "AwayProb":
                    "{:.1%}".format,
            },
        )
    )


if __name__ == "__main__":
    main()