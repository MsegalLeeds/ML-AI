from pathlib import Path
import joblib
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
MODEL_FILE = BASE_DIR / "premier_league_model.joblib"

VALIDATION_SEASON = "Season2024-2025"
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


def evaluate(
    model: Pipeline,
    dataset: pd.DataFrame,
    features: list[str],
    name: str,
) -> dict[str, float]:
    clean = dataset.dropna(
        subset=features + ["Target"]
    ).copy()

    probabilities = model.predict_proba(
        clean[features]
    )
    predictions = model.predict(
        clean[features]
    )

    accuracy = accuracy_score(
        clean["Target"],
        predictions,
    )
    loss = log_loss(
        clean["Target"],
        probabilities,
    )

    print()
    print(name)
    print("=" * 45)
    print(f"Rows: {len(clean)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Log loss: {loss:.3f}")

    return {
        "accuracy": accuracy,
        "log_loss": loss,
        "rows": len(clean),
    }


def train_candidate(
    train: pd.DataFrame,
    features: list[str],
) -> Pipeline:
    clean = train.dropna(
        subset=features + ["Target"]
    ).copy()

    model = make_model()
    model.fit(
        clean[features],
        clean["Target"],
    )
    return model


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    print("Available seasons:")
    print(df["Season"].unique())

    base_features = get_base_feature_columns()
    venue_features = get_venue_feature_columns()

    validation = df[
        df["Season"] == VALIDATION_SEASON
    ].copy()

    if validation.empty:
        raise ValueError(
            f"No rows found for {VALIDATION_SEASON}"
        )

    validation_start = validation["Date"].min()
    train_before_validation = df[
        df["Date"] < validation_start
    ].copy()

    # Evaluate both models on the SAME validation matches.
    validation_common = validation.dropna(
        subset=venue_features + ["Target"]
    ).copy()

    candidates = {
        "rolling_elo": base_features,
        "rolling_elo_venue": venue_features,
    }

    validation_results = {}

    for name, features in candidates.items():
        model = train_candidate(
            train_before_validation,
            features,
        )

        probabilities = model.predict_proba(
            validation_common[features]
        )
        predictions = model.predict(
            validation_common[features]
        )

        result = {
            "accuracy": accuracy_score(
                validation_common["Target"],
                predictions,
            ),
            "log_loss": log_loss(
                validation_common["Target"],
                probabilities,
            ),
        }

        validation_results[name] = result

        print()
        print(f"Validation: {name}")
        print("=" * 45)
        print(f"Rows: {len(validation_common)}")
        print(
            f"Accuracy: {result['accuracy']:.3f}"
        )
        print(
            f"Log loss: {result['log_loss']:.3f}"
        )

    selected_name = min(
        validation_results,
        key=lambda name: (
            validation_results[name]["log_loss"]
        ),
    )
    selected_features = candidates[selected_name]

    print()
    print(
        f"Selected model: {selected_name} "
        "(lowest validation log loss)"
    )

    # Honest test on 2025/26: refit using everything before the test
    # season, then evaluate once on the test season.
    test = df[
        df["Season"] == TEST_SEASON
    ].copy()

    if test.empty:
        raise ValueError(
            f"No rows found for {TEST_SEASON}"
        )

    test_start = test["Date"].min()
    train_before_test = df[
        df["Date"] < test_start
    ].copy()

    test_model = train_candidate(
        train_before_test,
        selected_features,
    )

    evaluate(
        test_model,
        test,
        selected_features,
        f"Final test {TEST_SEASON}: {selected_name}",
    )

    # Final production model: train on all completed historical/current
    # matches for which the selected features are available.
    final_training = df.dropna(
        subset=selected_features + ["Target"]
    ).copy()

    final_model = make_model()
    final_model.fit(
        final_training[selected_features],
        final_training["Target"],
    )

    artifact = {
        "model": final_model,
        "features": selected_features,
        "model_name": selected_name,
        "validation_results": validation_results,
        "trained_through": (
            final_training["Date"]
            .max()
            .isoformat()
        ),
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
    )

    print()
    print(f"Saved model: {MODEL_FILE}")
    print(
        f"Features saved with model: "
        f"{len(selected_features)}"
    )


if __name__ == "__main__":
    main()
