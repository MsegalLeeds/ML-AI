from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import (
    get_base_feature_columns,
    get_venue_feature_columns,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "model_data.csv"
TEST_SEASON = "Season2025-2026"


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
            ),
        ),
    ])


def multiclass_brier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    actual = np.zeros_like(probabilities)

    actual[
        np.arange(len(y_true)),
        y_true.astype(int),
    ] = 1

    return float(
        np.mean(
            np.sum(
                (probabilities - actual) ** 2,
                axis=1,
            )
        )
    )


def bookmaker_probabilities(
    df: pd.DataFrame,
) -> np.ndarray:
    raw = np.column_stack([
        1.0 / df["B365H"].astype(float),
        1.0 / df["B365D"].astype(float),
        1.0 / df["B365A"].astype(float),
    ])

    # Remove the bookmaker overround so rows sum to 1.
    return raw / raw.sum(
        axis=1,
        keepdims=True,
    )


def print_metrics(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    print()
    print(name)
    print("=" * 45)
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


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    base_features = get_base_feature_columns()
    venue_features = get_venue_feature_columns()

    test = df[
        df["Season"] == TEST_SEASON
    ].copy()

    if test.empty:
        raise ValueError(
            f"No rows found for {TEST_SEASON}"
        )

    # One common subset means changes in metrics come from the models,
    # not from evaluating different fixtures.
    required_test = venue_features + [
        "Target",
        "B365H",
        "B365D",
        "B365A",
    ]

    common_test = test.dropna(
        subset=required_test
    ).copy()

    test_start = test["Date"].min()

    train = df[
        df["Date"] < test_start
    ].copy()

    y_test = common_test[
        "Target"
    ].to_numpy()

    for name, features in [
        ("Rolling + Elo", base_features),
        (
            "Rolling + Elo + Venue",
            venue_features,
        ),
    ]:
        clean_train = train.dropna(
            subset=features + ["Target"]
        ).copy()

        model = make_model()
        model.fit(
            clean_train[features],
            clean_train["Target"],
        )

        probabilities = model.predict_proba(
            common_test[features]
        )

        print_metrics(
            name,
            y_test,
            probabilities,
        )

    market_probabilities = bookmaker_probabilities(
        common_test
    )

    print_metrics(
        "Bet365 Market",
        y_test,
        market_probabilities,
    )

    print()
    print(
        f"All three models evaluated on "
        f"{len(common_test)} identical matches."
    )


if __name__ == "__main__":
    main()
