from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = BASE_DIR / "model_data.csv"

OVERALL_WINDOW = 5
VENUE_WINDOW = 3

REST_DAYS_CAP = 14

RAW_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "B365H",
    "B365D",
    "B365A",
]

REQUIRED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
]


def load_all_seasons() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.csv"))

    print(f"Looking for CSV files in: {DATA_DIR}")
    print(f"Found {len(files)} files:")

    for file in files:
        print(f"  {file.name}")

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    frames = []

    for path in files:
        print(f"Loading {path.name}")

        # Read only the columns used by this project. This also avoids
        # carrying hundreds of bookmaker columns through the pipeline.
        df = pd.read_csv(
            path,
            usecols=lambda column: column in RAW_COLUMNS,
        ).copy()

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {missing}"
            )

        # Some historical files may not contain every Bet365 column.
        for odds_column in ["B365H", "B365D", "B365A"]:
            if odds_column not in df.columns:
                df[odds_column] = pd.NA

        df["Season"] = path.stem
        frames.append(df)

    matches = pd.concat(
        frames,
        ignore_index=True,
    )

    matches["Date"] = pd.to_datetime(
        matches["Date"],
        dayfirst=True,
        errors="coerce",
    )

    matches = matches.dropna(
        subset=[
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "FTHG",
            "FTAG",
        ]
    ).copy()

    matches = matches[
        matches["FTR"].isin(["H", "D", "A"])
    ].copy()

    return (
        matches
        .sort_values(["Date", "HomeTeam", "AwayTeam"])
        .reset_index(drop=True)
    )


def _elo_step(
    home_elo: float,
    away_elo: float,
    result: str,
    k_factor: float,
    home_advantage: float,
) -> tuple[float, float]:
    adjusted_home = home_elo + home_advantage

    expected_home = 1.0 / (
        1.0
        + 10.0 ** (
            (away_elo - adjusted_home) / 400.0
        )
    )
    expected_away = 1.0 - expected_home

    if result == "H":
        actual_home, actual_away = 1.0, 0.0
    elif result == "D":
        actual_home, actual_away = 0.5, 0.5
    elif result == "A":
        actual_home, actual_away = 0.0, 1.0
    else:
        raise ValueError(f"Unexpected result: {result}")

    return (
        home_elo
        + k_factor * (actual_home - expected_home),
        away_elo
        + k_factor * (actual_away - expected_away),
    )


def add_elo_ratings(
    matches: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 30.0,
    home_advantage: float = 100.0,
) -> pd.DataFrame:
    """
    Add PRE-MATCH Elo ratings.

    Elo carries across seasons. The current match result is used only
    after its pre-match ratings have been recorded.
    """
    df = matches.sort_values("Date").copy()

    ratings: dict[str, float] = {}
    home_elos: list[float] = []
    away_elos: list[float] = []

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_elo = ratings.get(home_team, initial_rating)
        away_elo = ratings.get(away_team, initial_rating)

        home_elos.append(home_elo)
        away_elos.append(away_elo)

        new_home, new_away = _elo_step(
            home_elo=home_elo,
            away_elo=away_elo,
            result=row["FTR"],
            k_factor=k_factor,
            home_advantage=home_advantage,
        )

        ratings[home_team] = new_home
        ratings[away_team] = new_away

    df["HomeElo"] = home_elos
    df["AwayElo"] = away_elos
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]

    return df


def calculate_current_elo_ratings(
    matches: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 30.0,
    home_advantage: float = 100.0,
) -> dict[str, float]:
    """Return team Elo ratings AFTER all completed matches."""
    ratings: dict[str, float] = {}

    for _, row in matches.sort_values("Date").iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_elo = ratings.get(home_team, initial_rating)
        away_elo = ratings.get(away_team, initial_rating)

        new_home, new_away = _elo_step(
            home_elo=home_elo,
            away_elo=away_elo,
            result=row["FTR"],
            k_factor=k_factor,
            home_advantage=home_advantage,
        )

        ratings[home_team] = new_home
        ratings[away_team] = new_away

    return ratings


def build_team_history(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[
        [
            "Date",
            "Season",
            "HomeTeam",
            "FTHG",
            "FTAG",
            "HS",
            "AS",
            "HST",
            "AST",
            "FTR",
            "HomeElo",
            "AwayElo",
            "HomeRestDays",
            "AwayRestDays",
            "RestDaysDiff",
        ]
    ].copy()

    home = home.rename(
        columns={
            "HomeTeam": "Team",
            "FTHG": "GoalsFor",
            "FTAG": "GoalsAgainst",
            "HS": "ShotsFor",
            "AS": "ShotsAgainst",
            "HST": "ShotsOnTargetFor",
            "AST": "ShotsOnTargetAgainst",
            "HomeElo": "TeamElo",
            "AwayElo": "OpponentElo",
            "HomeRestDays": "HomeRestDays",
            "AwayRestDays": "AwayRestDays",
            "RestDaysDiff": "RestDaysDiff",
        }
    )

    home["Venue"] = "Home"
    home["Points"] = home["FTR"].map(
        {"H": 3, "D": 1, "A": 0}
    )

    away = matches[
        [
            "Date",
            "Season",
            "AwayTeam",
            "FTAG",
            "FTHG",
            "AS",
            "HS",
            "AST",
            "HST",
            "FTR",
            "HomeElo",
            "AwayElo",
            "HomeRestDays",
            "AwayRestDays",
            "RestDaysDiff",
        ]
    ].copy()

    away = away.rename(
        columns={
            "AwayTeam": "Team",
            "FTAG": "GoalsFor",
            "FTHG": "GoalsAgainst",
            "AS": "ShotsFor",
            "HS": "ShotsAgainst",
            "AST": "ShotsOnTargetFor",
            "HST": "ShotsOnTargetAgainst",
            "HomeElo": "OpponentElo",
            "AwayElo": "TeamElo",
            "HomeRestDays": "HomeRestDays",
            "AwayRestDays": "AwayRestDays",
            "RestDaysDiff": "RestDaysDiff",
        }
    )

    away["Venue"] = "Away"
    away["Points"] = away["FTR"].map(
        {"A": 3, "D": 1, "H": 0}
    )

    team_history = pd.concat(
        [home, away],
        ignore_index=True,
    )

    team_history["EloActual"] = (
        team_history["Points"]
        .map({
            3: 1.0,
            1: 0.5,
            0: 0.0,
        })
    )

    team_advantage = np.where(
        team_history["Venue"] == "Home",
        100.0,
        0.0,
    )

    opponent_advantage = np.where(
        team_history["Venue"] == "Away",
        100.0,
        0.0,
    )

    team_history["ExpectedEloScore"] = (
            1.0
            /
            (
                    1.0
                    + 10.0 ** (
                            (
                                    team_history["OpponentElo"]
                                    + opponent_advantage
                                    - team_history["TeamElo"]
                                    - team_advantage
                            )
                            / 400.0
                    )
            )
    )

    team_history["EloOverperformance"] = (
            team_history["EloActual"]
            - team_history["ExpectedEloScore"]
    )

    return (
        team_history
        .sort_values(["Season", "Team", "Date"])
        .reset_index(drop=True)
    )


def add_rolling_features(
    team_history: pd.DataFrame,
    window: int = OVERALL_WINDOW,
) -> pd.DataFrame:
    """
    Overall form within the current Premier League season.

    shift(1) prevents the current result from leaking into its own
    prediction features.
    """
    df = team_history.copy()

    stats = [
        "GoalsFor",
        "GoalsAgainst",
        "ShotsFor",
        "ShotsAgainst",
        "ShotsOnTargetFor",
        "ShotsOnTargetAgainst",
        "Points",
        "OpponentElo",
        "EloOverperformance",
    ]

    for stat in stats:
        df[f"{stat}_last{window}"] = (
            df.groupby(["Season", "Team"])[stat]
            .transform(
                lambda values: (
                    values
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
        )

    return df


def add_venue_features(
    team_history: pd.DataFrame,
    window: int = VENUE_WINDOW,
) -> pd.DataFrame:
    """
    Venue form within the current season.

    A 3-game venue window is used so current-season predictions become
    available earlier than they would with five home/away matches.
    """
    df = team_history.copy()

    stats = [
        "GoalsFor",
        "GoalsAgainst",
        "ShotsFor",
        "ShotsAgainst",
        "ShotsOnTargetFor",
        "ShotsOnTargetAgainst",
        "Points",
    ]

    for stat in stats:
        df[f"{stat}_venue_last{window}"] = (
            df.groupby(
                ["Season", "Team", "Venue"]
            )[stat]
            .transform(
                lambda values: (
                    values
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
        )

    return df


def get_base_feature_columns(
    overall_window: int = OVERALL_WINDOW,
) -> list[str]:
    return [
        f"Home_GoalsFor_last{overall_window}",
        f"Home_GoalsAgainst_last{overall_window}",
        f"Home_ShotsFor_last{overall_window}",
        f"Home_ShotsAgainst_last{overall_window}",
        f"Home_ShotsOnTargetFor_last{overall_window}",
        f"Home_ShotsOnTargetAgainst_last{overall_window}",
        f"Home_Points_last{overall_window}",
        f"Home_OpponentElo_last{overall_window}",
        f"Home_EloOverperformance_last{overall_window}",

        f"Away_GoalsFor_last{overall_window}",
        f"Away_GoalsAgainst_last{overall_window}",
        f"Away_ShotsFor_last{overall_window}",
        f"Away_ShotsAgainst_last{overall_window}",
        f"Away_ShotsOnTargetFor_last{overall_window}",
        f"Away_ShotsOnTargetAgainst_last{overall_window}",
        f"Away_Points_last{overall_window}",
        f"Away_OpponentElo_last{overall_window}",
        f"Away_EloOverperformance_last{overall_window}",

        "GoalsForDiff",
        "GoalsAgainstDiff",
        "ShotsDiff",
        "ShotsOnTargetDiff",
        "PointsDiff",
        "EloDiff",
        "OpponentStrengthDiff",
        "EloFormDiff",
        "HomeRestDays",
        "AwayRestDays",
        "RestDaysDiff",
    ]


def get_venue_feature_columns(
    overall_window: int = OVERALL_WINDOW,
    venue_window: int = VENUE_WINDOW,
) -> list[str]:
    return get_base_feature_columns(overall_window) + [
        f"Home_GoalsFor_venue_last{venue_window}",
        f"Home_GoalsAgainst_venue_last{venue_window}",
        f"Home_ShotsFor_venue_last{venue_window}",
        f"Home_ShotsAgainst_venue_last{venue_window}",
        f"Home_ShotsOnTargetFor_venue_last{venue_window}",
        f"Home_ShotsOnTargetAgainst_venue_last{venue_window}",
        f"Home_Points_venue_last{venue_window}",

        f"Away_GoalsFor_venue_last{venue_window}",
        f"Away_GoalsAgainst_venue_last{venue_window}",
        f"Away_ShotsFor_venue_last{venue_window}",
        f"Away_ShotsAgainst_venue_last{venue_window}",
        f"Away_ShotsOnTargetFor_venue_last{venue_window}",
        f"Away_ShotsOnTargetAgainst_venue_last{venue_window}",
        f"Away_Points_venue_last{venue_window}",

        "VenueGoalsForDiff",
        "VenueGoalsAgainstDiff",
        "VenueShotsDiff",
        "VenueShotsOnTargetDiff",
        "VenuePointsDiff",
    ]


def build_model_dataset(
    matches: pd.DataFrame,
    team_history: pd.DataFrame,
    overall_window: int = OVERALL_WINDOW,
    venue_window: int = VENUE_WINDOW,
) -> pd.DataFrame:
    overall_cols = [
        f"GoalsFor_last{overall_window}",
        f"GoalsAgainst_last{overall_window}",
        f"ShotsFor_last{overall_window}",
        f"ShotsAgainst_last{overall_window}",
        f"ShotsOnTargetFor_last{overall_window}",
        f"ShotsOnTargetAgainst_last{overall_window}",
        f"Points_last{overall_window}",
        f"OpponentElo_last{overall_window}",
        f"EloOverperformance_last{overall_window}",
    ]

    venue_cols = [
        f"GoalsFor_venue_last{venue_window}",
        f"GoalsAgainst_venue_last{venue_window}",
        f"ShotsFor_venue_last{venue_window}",
        f"ShotsAgainst_venue_last{venue_window}",
        f"ShotsOnTargetFor_venue_last{venue_window}",
        f"ShotsOnTargetAgainst_venue_last{venue_window}",
        f"Points_venue_last{venue_window}",
    ]

    all_form_cols = overall_cols + venue_cols

    home_features = team_history[
        ["Date", "Season", "Team", "Venue"]
        + all_form_cols
    ].copy()

    home_features = (
        home_features[
            home_features["Venue"] == "Home"
        ]
        .drop(columns="Venue")
        .rename(
            columns={
                "Team": "HomeTeam",
                **{
                    col: f"Home_{col}"
                    for col in all_form_cols
                },
            }
        )
    )

    away_features = team_history[
        ["Date", "Season", "Team", "Venue"]
        + all_form_cols
    ].copy()

    away_features = (
        away_features[
            away_features["Venue"] == "Away"
        ]
        .drop(columns="Venue")
        .rename(
            columns={
                "Team": "AwayTeam",
                **{
                    col: f"Away_{col}"
                    for col in all_form_cols
                },
            }
        )
    )

    model_df = matches.merge(
        home_features,
        on=["Date", "Season", "HomeTeam"],
        how="left",
        validate="one_to_one",
    )

    model_df = model_df.merge(
        away_features,
        on=["Date", "Season", "AwayTeam"],
        how="left",
        validate="one_to_one",
    )

    model_df["Target"] = model_df["FTR"].map(
        {"H": 0, "D": 1, "A": 2}
    )

    model_df["GoalsForDiff"] = (
        model_df[
            f"Home_GoalsFor_last{overall_window}"
        ]
        - model_df[
            f"Away_GoalsFor_last{overall_window}"
        ]
    )

    model_df["GoalsAgainstDiff"] = (
        model_df[
            f"Home_GoalsAgainst_last{overall_window}"
        ]
        - model_df[
            f"Away_GoalsAgainst_last{overall_window}"
        ]
    )

    model_df["ShotsDiff"] = (
        model_df[
            f"Home_ShotsFor_last{overall_window}"
        ]
        - model_df[
            f"Away_ShotsFor_last{overall_window}"
        ]
    )

    model_df["ShotsOnTargetDiff"] = (
        model_df[
            f"Home_ShotsOnTargetFor_last{overall_window}"
        ]
        - model_df[
            f"Away_ShotsOnTargetFor_last{overall_window}"
        ]
    )

    model_df["PointsDiff"] = (
        model_df[
            f"Home_Points_last{overall_window}"
        ]
        - model_df[
            f"Away_Points_last{overall_window}"
        ]
    )

    model_df["VenueGoalsForDiff"] = (
        model_df[
            f"Home_GoalsFor_venue_last{venue_window}"
        ]
        - model_df[
            f"Away_GoalsFor_venue_last{venue_window}"
        ]
    )

    model_df["VenueGoalsAgainstDiff"] = (
        model_df[
            f"Home_GoalsAgainst_venue_last{venue_window}"
        ]
        - model_df[
            f"Away_GoalsAgainst_venue_last{venue_window}"
        ]
    )

    model_df["VenueShotsDiff"] = (
        model_df[
            f"Home_ShotsFor_venue_last{venue_window}"
        ]
        - model_df[
            f"Away_ShotsFor_venue_last{venue_window}"
        ]
    )

    model_df["VenueShotsOnTargetDiff"] = (
        model_df[
            f"Home_ShotsOnTargetFor_venue_last{venue_window}"
        ]
        - model_df[
            f"Away_ShotsOnTargetFor_venue_last{venue_window}"
        ]
    )

    model_df["VenuePointsDiff"] = (
        model_df[
            f"Home_Points_venue_last{venue_window}"
        ]
        - model_df[
            f"Away_Points_venue_last{venue_window}"
        ]
    )

    model_df["OpponentStrengthDiff"] = (
            model_df[
                f"Home_OpponentElo_last{overall_window}"
            ]
            - model_df[
                f"Away_OpponentElo_last{overall_window}"
            ]
    )

    model_df["EloFormDiff"] = (
            model_df[
                f"Home_EloOverperformance_last{overall_window}"
            ]
            - model_df[
                f"Away_EloOverperformance_last{overall_window}"
            ]
    )

    # EloDiff was created before the merges, but recalculate it to make
    # the relationship explicit.
    model_df["EloDiff"] = (
        model_df["HomeElo"]
        - model_df["AwayElo"]
    )

    return (
        model_df
        .sort_values("Date")
        .reset_index(drop=True)
    )

def add_rest_days(matches):
    """
    Adds pre-match rest days for each team.

    Only previous completed Premier League matches are used,
    so there is no target leakage.

    Rest is capped at 14 days so summer/international breaks
    do not create huge values.
    """

    matches = matches.sort_values("Date").copy()

    last_match = {}

    home_rest = []
    away_rest = []

    for _, row in matches.iterrows():

        date = pd.Timestamp(row["Date"])
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        if home in last_match:
            rest = (
                date - last_match[home]
            ).days

            home_rest.append(
                min(rest, REST_DAYS_CAP)
            )
        else:
            home_rest.append(np.nan)

        if away in last_match:
            rest = (
                date - last_match[away]
            ).days

            away_rest.append(
                min(rest, REST_DAYS_CAP)
            )
        else:
            away_rest.append(np.nan)

        last_match[home] = date
        last_match[away] = date

    matches["HomeRestDays"] = (
        home_rest
    )

    matches["AwayRestDays"] = (
        away_rest
    )

    matches["RestDaysDiff"] = (
        matches["HomeRestDays"]
        - matches["AwayRestDays"]
    )

    return matches

def test_rest_days():
    matches = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01",
            "2026-01-05",
            "2026-01-08",
        ]),
        "HomeTeam": [
            "Arsenal",
            "Chelsea",
            "Arsenal",
        ],
        "AwayTeam": [
            "Chelsea",
            "Liverpool",
            "Liverpool",
        ],
    })

    result = add_rest_days(
        matches
    )

    arsenal_second_match = (
        result.iloc[2]
    )

    assert (
        arsenal_second_match[
            "HomeRestDays"
        ]
        == 7
    )


def main() -> None:
    matches = load_all_seasons()
    matches = add_rest_days(
        matches
    )
    matches = add_elo_ratings(matches)

    team_history = build_team_history(matches)
    team_history = add_rolling_features(team_history)
    team_history = add_venue_features(team_history)

    model_df = build_model_dataset(
        matches,
        team_history,
    )

    # Do NOT drop early-season rows here. Different candidate models use
    # different feature sets, so train_model.py/benchmark.py choose the
    # appropriate complete rows themselves.
    model_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    base_features = get_base_feature_columns()
    venue_features = get_venue_feature_columns()

    base_rows = model_df.dropna(
        subset=base_features + ["Target"]
    )
    venue_rows = model_df.dropna(
        subset=venue_features + ["Target"]
    )

    print()
    print(f"Total completed matches: {len(model_df)}")
    print(f"Rolling + Elo usable rows: {len(base_rows)}")
    print(
        "Rolling + Elo + Venue usable rows: "
        f"{len(venue_rows)}"
    )
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
