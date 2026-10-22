from difflib import get_close_matches
from pathlib import Path

import joblib
import pandas as pd

from prepare_data import (
    OVERALL_WINDOW,
    VENUE_WINDOW,
    build_team_history,
    calculate_current_elo_ratings,
    load_all_seasons,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "premier_league_model.joblib"


def resolve_team_name(
    requested: str,
    available_teams: list[str],
) -> str:
    exact = {
        team.lower(): team
        for team in available_teams
    }

    key = requested.strip().lower()

    if key in exact:
        return exact[key]

    lower_names = list(exact.keys())
    matches = get_close_matches(
        key,
        lower_names,
        n=3,
        cutoff=0.6,
    )

    if matches:
        suggestions = [
            exact[name]
            for name in matches
        ]

        raise ValueError(
            f"Unknown team '{requested}'. "
            f"Did you mean: {', '.join(suggestions)}?"
        )

    raise ValueError(
        f"Unknown team '{requested}'."
    )


def current_season_name(
    matches: pd.DataFrame,
) -> str:
    latest_row = (
        matches
        .sort_values("Date")
        .iloc[-1]
    )
    return str(latest_row["Season"])


def get_team_form(
    team_history: pd.DataFrame,
    season: str,
    team: str,
    venue: str,
) -> dict[str, float]:
    season_history = team_history[
        (team_history["Season"] == season)
        & (team_history["Team"] == team)
    ].sort_values("Date")

    overall = season_history.tail(
        OVERALL_WINDOW
    )

    if len(overall) < OVERALL_WINDOW:
        raise ValueError(
            f"{team} has only {len(overall)} completed "
            f"{season} matches. This model requires "
            f"{OVERALL_WINDOW} current-season matches."
        )

    venue_history = season_history[
        season_history["Venue"] == venue
    ].tail(VENUE_WINDOW)

    return {
        "GoalsFor":
            float(overall["GoalsFor"].mean()),
        "GoalsAgainst":
            float(overall["GoalsAgainst"].mean()),
        "ShotsFor":
            float(overall["ShotsFor"].mean()),
        "ShotsAgainst":
            float(overall["ShotsAgainst"].mean()),
        "ShotsOnTargetFor":
            float(
                overall[
                    "ShotsOnTargetFor"
                ].mean()
            ),
        "ShotsOnTargetAgainst":
            float(
                overall[
                    "ShotsOnTargetAgainst"
                ].mean()
            ),
        "Points":
            float(overall["Points"].mean()),

        # Venue fields are None until enough current-season home/away
        # games exist. They are only required if the saved model selected
        # venue features.
        "VenueGoalsFor":
            (
                float(
                    venue_history[
                        "GoalsFor"
                    ].mean()
                )
                if len(venue_history) >= VENUE_WINDOW
                else None
            ),
        "VenueGoalsAgainst":
            (
                float(
                    venue_history[
                        "GoalsAgainst"
                    ].mean()
                )
                if len(venue_history) >= VENUE_WINDOW
                else None
            ),
        "VenueShotsFor":
            (
                float(
                    venue_history[
                        "ShotsFor"
                    ].mean()
                )
                if len(venue_history) >= VENUE_WINDOW
                else None
            ),
        "VenueShotsAgainst":
            (
                float(
                    venue_history[
                        "ShotsAgainst"
                    ].mean()
                )
                if len(venue_history) >= VENUE_WINDOW
                else None
            ),
        "VenueShotsOnTargetFor":
            (
                float(
                    venue_history[
                        "ShotsOnTargetFor"
                    ].mean()
                )
                if len(venue_history) >= VENUE_WINDOW
                else None
            ),
        "VenueShotsOnTargetAgainst":
            (
                float(
                    venue_history[
                        "ShotsOnTargetAgainst"
                    ].mean()
                )
                if len(venue_history) >= VENUE_WINDOW
                else None
            ),
        "VenuePoints":
            (
                float(
                    venue_history[
                        "Points"
                    ].mean()
                )
                if len(venue_history) >= VENUE_WINDOW
                else None
            ),
        "VenueMatches":
            len(venue_history),
    }


def build_prediction_row(
    home: dict[str, float],
    away: dict[str, float],
    home_elo: float,
    away_elo: float,
) -> dict[str, float | None]:
    row = {
        f"Home_GoalsFor_last{OVERALL_WINDOW}":
            home["GoalsFor"],
        f"Home_GoalsAgainst_last{OVERALL_WINDOW}":
            home["GoalsAgainst"],
        f"Home_ShotsFor_last{OVERALL_WINDOW}":
            home["ShotsFor"],
        f"Home_ShotsAgainst_last{OVERALL_WINDOW}":
            home["ShotsAgainst"],
        f"Home_ShotsOnTargetFor_last{OVERALL_WINDOW}":
            home["ShotsOnTargetFor"],
        f"Home_ShotsOnTargetAgainst_last{OVERALL_WINDOW}":
            home["ShotsOnTargetAgainst"],
        f"Home_Points_last{OVERALL_WINDOW}":
            home["Points"],

        f"Away_GoalsFor_last{OVERALL_WINDOW}":
            away["GoalsFor"],
        f"Away_GoalsAgainst_last{OVERALL_WINDOW}":
            away["GoalsAgainst"],
        f"Away_ShotsFor_last{OVERALL_WINDOW}":
            away["ShotsFor"],
        f"Away_ShotsAgainst_last{OVERALL_WINDOW}":
            away["ShotsAgainst"],
        f"Away_ShotsOnTargetFor_last{OVERALL_WINDOW}":
            away["ShotsOnTargetFor"],
        f"Away_ShotsOnTargetAgainst_last{OVERALL_WINDOW}":
            away["ShotsOnTargetAgainst"],
        f"Away_Points_last{OVERALL_WINDOW}":
            away["Points"],

        "GoalsForDiff":
            home["GoalsFor"]
            - away["GoalsFor"],
        "GoalsAgainstDiff":
            home["GoalsAgainst"]
            - away["GoalsAgainst"],
        "ShotsDiff":
            home["ShotsFor"]
            - away["ShotsFor"],
        "ShotsOnTargetDiff":
            home["ShotsOnTargetFor"]
            - away["ShotsOnTargetFor"],
        "PointsDiff":
            home["Points"]
            - away["Points"],

        "EloDiff":
            home_elo - away_elo,
    }

    # Add venue features even if the selected model does not use them.
    # The saved feature list decides which columns are passed to sklearn.
    row.update({
        f"Home_GoalsFor_venue_last{VENUE_WINDOW}":
            home["VenueGoalsFor"],
        f"Home_GoalsAgainst_venue_last{VENUE_WINDOW}":
            home["VenueGoalsAgainst"],
        f"Home_ShotsFor_venue_last{VENUE_WINDOW}":
            home["VenueShotsFor"],
        f"Home_ShotsAgainst_venue_last{VENUE_WINDOW}":
            home["VenueShotsAgainst"],
        f"Home_ShotsOnTargetFor_venue_last{VENUE_WINDOW}":
            home["VenueShotsOnTargetFor"],
        f"Home_ShotsOnTargetAgainst_venue_last{VENUE_WINDOW}":
            home["VenueShotsOnTargetAgainst"],
        f"Home_Points_venue_last{VENUE_WINDOW}":
            home["VenuePoints"],

        f"Away_GoalsFor_venue_last{VENUE_WINDOW}":
            away["VenueGoalsFor"],
        f"Away_GoalsAgainst_venue_last{VENUE_WINDOW}":
            away["VenueGoalsAgainst"],
        f"Away_ShotsFor_venue_last{VENUE_WINDOW}":
            away["VenueShotsFor"],
        f"Away_ShotsAgainst_venue_last{VENUE_WINDOW}":
            away["VenueShotsAgainst"],
        f"Away_ShotsOnTargetFor_venue_last{VENUE_WINDOW}":
            away["VenueShotsOnTargetFor"],
        f"Away_ShotsOnTargetAgainst_venue_last{VENUE_WINDOW}":
            away["VenueShotsOnTargetAgainst"],
        f"Away_Points_venue_last{VENUE_WINDOW}":
            away["VenuePoints"],
    })

    if (
        home["VenueGoalsFor"] is not None
        and away["VenueGoalsFor"] is not None
    ):
        row["VenueGoalsForDiff"] = (
            home["VenueGoalsFor"]
            - away["VenueGoalsFor"]
        )
        row["VenueGoalsAgainstDiff"] = (
            home["VenueGoalsAgainst"]
            - away["VenueGoalsAgainst"]
        )
        row["VenueShotsDiff"] = (
            home["VenueShotsFor"]
            - away["VenueShotsFor"]
        )
        row["VenueShotsOnTargetDiff"] = (
            home["VenueShotsOnTargetFor"]
            - away["VenueShotsOnTargetFor"]
        )
        row["VenuePointsDiff"] = (
            home["VenuePoints"]
            - away["VenuePoints"]
        )
    else:
        row["VenueGoalsForDiff"] = None
        row["VenueGoalsAgainstDiff"] = None
        row["VenueShotsDiff"] = None
        row["VenueShotsOnTargetDiff"] = None
        row["VenuePointsDiff"] = None

    return row


def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "premier_league_model.joblib was not found. "
            "Run prepare_data.py and train_model.py first."
        )

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    features = saved["features"]
    model_name = saved.get(
        "model_name",
        "unknown",
    )

    matches = load_all_seasons()
    season = current_season_name(matches)

    team_history = build_team_history(matches)
    available_teams = sorted(
        team_history[
            team_history["Season"] == season
        ]["Team"].unique()
    )

    current_elos = calculate_current_elo_ratings(
        matches
    )

    print()
    print("Premier League Predictor")
    print("=" * 40)
    print(f"Model: {model_name}")
    print(f"Current season: {season}")
    print()

    requested_home = input(
        "Home team: "
    ).strip()

    requested_away = input(
        "Away team: "
    ).strip()

    home_team = resolve_team_name(
        requested_home,
        available_teams,
    )
    away_team = resolve_team_name(
        requested_away,
        available_teams,
    )

    home = get_team_form(
        team_history,
        season,
        home_team,
        "Home",
    )

    away = get_team_form(
        team_history,
        season,
        away_team,
        "Away",
    )

    row = build_prediction_row(
        home=home,
        away=away,
        home_elo=current_elos.get(
            home_team,
            1500.0,
        ),
        away_elo=current_elos.get(
            away_team,
            1500.0,
        ),
    )

    X = pd.DataFrame(
        [row]
    ).reindex(
        columns=features
    )

    missing = X.columns[
        X.isna().any()
    ].tolist()

    if missing:
        venue_features_missing = [
            feature
            for feature in missing
            if "venue" in feature.lower()
            or feature.startswith("Venue")
        ]

        if venue_features_missing:
            raise ValueError(
                "The saved model selected venue features, but "
                "there are not yet enough current-season venue "
                "matches for this fixture. "
                f"{home_team} has "
                f"{home['VenueMatches']} usable home matches and "
                f"{away_team} has "
                f"{away['VenueMatches']} usable away matches; "
                f"{VENUE_WINDOW} are required."
            )

        raise ValueError(
            f"Missing prediction features: {missing}"
        )

    probabilities = model.predict_proba(X)[0]

    # Class ordering is stored by sklearn; do not assume it.
    classifier = model.named_steps[
        "classifier"
    ]
    probability_by_class = dict(
        zip(
            classifier.classes_,
            probabilities,
        )
    )

    home_probability = probability_by_class[0]
    draw_probability = probability_by_class[1]
    away_probability = probability_by_class[2]

    print()
    print(
        f"{home_team} vs {away_team}"
    )
    print("-" * 40)
    print(
        f"{home_team} win: "
        f"{home_probability:.1%}"
    )
    print(
        f"Draw: "
        f"{draw_probability:.1%}"
    )
    print(
        f"{away_team} win: "
        f"{away_probability:.1%}"
    )


if __name__ == "__main__":
    main()
