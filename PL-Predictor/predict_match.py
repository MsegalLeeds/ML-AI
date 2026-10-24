from difflib import get_close_matches
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import poisson

from prepare_data import (
    OVERALL_WINDOW,
    add_elo_ratings,
    add_rest_days,
    build_team_history,
    calculate_current_elo_ratings,
    load_all_seasons,
    REST_DAYS_CAP
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "premier_league_model.joblib"
)

MAX_GOALS = 10


TEAM_ALIASES = {
    "Man Utd": "Man United",
    "Manchester United": "Man United",

    "Manchester City": "Man City",

    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",

    "Nottingham Forest": "Nott'm Forest",

    "Brighton & Hove Albion": "Brighton",

    "AFC Bournemouth": "Bournemouth",

    "Ipswich Town": "Ipswich",

    "Coventry City": "Coventry",

    "Hull City": "Hull",

    "Leeds United": "Leeds",

    "Newcastle United": "Newcastle",

    "Sunderland AFC": "Sunderland",
}


class PremierLeaguePredictor:

    def __init__(self):
        saved = joblib.load(
            MODEL_FILE
        )

        self.model = (
            saved["logistic_model"]
        )

        self.home_goal_model = (
            saved["home_goal_model"]
        )

        self.away_goal_model = (
            saved["away_goal_model"]
        )

        self.temperature = (
            saved["temperature"]
        )

        self.features = (
            saved["features"]
        )

        self.model_name = (
            saved["model_name"]
        )

        self.matches = load_all_seasons()

        self.matches = add_rest_days(
            self.matches
        )

        self.matches = add_elo_ratings(
            self.matches
        )

        self.season = (
            self.matches
            .sort_values("Date")
            .iloc[-1]["Season"]
        )

        self.team_history = (
            build_team_history(
                self.matches
            )
        )

        self.elos = (
            calculate_current_elo_ratings(
                self.matches
            )
        )

        self.teams = sorted(
            self.team_history[
                self.team_history["Season"]
                == self.season
            ]["Team"].unique()
        )

        self._form_cache = {}

    def normalise_team(
        self,
        team,
    ):
        team = TEAM_ALIASES.get(
            team,
            team,
        )

        exact = {
            x.lower(): x
            for x in self.teams
        }

        key = team.lower()

        if key in exact:
            return exact[key]

        matches = get_close_matches(
            key,
            list(exact),
            n=3,
            cutoff=0.6,
        )

        if matches:
            suggestions = [
                exact[x]
                for x in matches
            ]

            raise ValueError(
                f"Unknown team '{team}'. "
                f"Did you mean: "
                f"{', '.join(suggestions)}?"
            )

        raise ValueError(
            f"Unknown team '{team}'"
        )

    def _form(
        self,
        team,
    ):
        if team in self._form_cache:
            return self._form_cache[
                team
            ]

        history = self.team_history[
            (
                self.team_history["Season"]
                == self.season
            )
            &
            (
                self.team_history["Team"]
                == team
            )
        ].sort_values("Date")

        latest = history.tail(
            OVERALL_WINDOW
        )

        if len(latest) < OVERALL_WINDOW:
            raise ValueError(
                f"{team} only has "
                f"{len(latest)} completed "
                f"current-season matches."
            )

        form = {
            "GoalsFor":
                latest[
                    "GoalsFor"
                ].mean(),

            "GoalsAgainst":
                latest[
                    "GoalsAgainst"
                ].mean(),

            "ShotsFor":
                latest[
                    "ShotsFor"
                ].mean(),

            "ShotsAgainst":
                latest[
                    "ShotsAgainst"
                ].mean(),

            "ShotsOnTargetFor":
                latest[
                    "ShotsOnTargetFor"
                ].mean(),

            "ShotsOnTargetAgainst":
                latest[
                    "ShotsOnTargetAgainst"
                ].mean(),

            "Points":
                latest[
                    "Points"
                ].mean(),

            "OpponentElo":
                latest[
                    "OpponentElo"
                ].mean(),

            "EloOverperformance":
                latest[
                    "EloOverperformance"
                ].mean(),
        }

        self._form_cache[team] = form

        return form

    def _feature_row(
        self,
        home_team,
        away_team,
        match_date,
        home_rest_days=None,
        away_rest_days=None,
    ):
        home = self._form(
            home_team
        )

        away = self._form(
            away_team
        )

        match_date = pd.Timestamp(
            match_date
        )

        if home_rest_days is None:
            home_matches = self.matches[
                self.matches["HomeTeam"].eq(home_team)
                |
                self.matches["AwayTeam"].eq(home_team)
                ]

            last_home_date = (
                home_matches["Date"].max()
            )

            home_rest_days = min(
                (
                        match_date
                        - last_home_date
                ).days,
                REST_DAYS_CAP,
            )

        if away_rest_days is None:
            away_matches = self.matches[
                self.matches["HomeTeam"].eq(away_team)
                |
                self.matches["AwayTeam"].eq(away_team)
                ]

            last_away_date = (
                away_matches["Date"].max()
            )

            away_rest_days = min(
                (
                        match_date
                        - last_away_date
                ).days,
                REST_DAYS_CAP,
            )

        row = {
            "Home_GoalsFor_last5":
                home["GoalsFor"],

            "Home_GoalsAgainst_last5":
                home["GoalsAgainst"],

            "Home_ShotsFor_last5":
                home["ShotsFor"],

            "Home_ShotsAgainst_last5":
                home["ShotsAgainst"],

            "Home_ShotsOnTargetFor_last5":
                home["ShotsOnTargetFor"],

            "Home_ShotsOnTargetAgainst_last5":
                home["ShotsOnTargetAgainst"],

            "Home_Points_last5":
                home["Points"],

            f"Home_OpponentElo_last{OVERALL_WINDOW}":
                home["OpponentElo"],

            f"Home_EloOverperformance_last{OVERALL_WINDOW}":
                home["EloOverperformance"],


            "Away_GoalsFor_last5":
                away["GoalsFor"],

            "Away_GoalsAgainst_last5":
                away["GoalsAgainst"],

            "Away_ShotsFor_last5":
                away["ShotsFor"],

            "Away_ShotsAgainst_last5":
                away["ShotsAgainst"],

            "Away_ShotsOnTargetFor_last5":
                away["ShotsOnTargetFor"],

            "Away_ShotsOnTargetAgainst_last5":
                away["ShotsOnTargetAgainst"],

            "Away_Points_last5":
                away["Points"],

            f"Away_OpponentElo_last{OVERALL_WINDOW}":
                away["OpponentElo"],

            f"Away_EloOverperformance_last{OVERALL_WINDOW}":
                away["EloOverperformance"],
        }

        row["GoalsForDiff"] = (
            home["GoalsFor"]
            - away["GoalsFor"]
        )

        row["GoalsAgainstDiff"] = (
            home["GoalsAgainst"]
            - away["GoalsAgainst"]
        )

        row["ShotsDiff"] = (
            home["ShotsFor"]
            - away["ShotsFor"]
        )

        row["ShotsOnTargetDiff"] = (
            home["ShotsOnTargetFor"]
            - away["ShotsOnTargetFor"]
        )

        row["PointsDiff"] = (
            home["Points"]
            - away["Points"]
        )

        row["EloDiff"] = (
            self.elos.get(
                home_team,
                1500.0,
            )
            -
            self.elos.get(
                away_team,
                1500.0,
            )
        )

        row["OpponentStrengthDiff"] = (
                home["OpponentElo"]
                - away["OpponentElo"]
        )

        row["EloFormDiff"] = (
                home["EloOverperformance"]
                - away["EloOverperformance"]
        )

        row["HomeRestDays"] = (
            home_rest_days
        )

        row["AwayRestDays"] = (
            away_rest_days
        )

        row["RestDaysDiff"] = (
                home_rest_days
                - away_rest_days
        )

        return pd.DataFrame(
            [row],
            columns=self.features,
        )

    def _temperature_scale(
        self,
        probabilities,
    ):
        probabilities = np.clip(
            probabilities,
            1e-12,
            1.0,
        )

        logits = (
            np.log(probabilities)
            / self.temperature
        )

        logits -= logits.max()

        exp_logits = np.exp(logits)

        return (
            exp_logits
            / exp_logits.sum()
        )

    def _score_matrix(
        self,
        expected_home,
        expected_away,
        result_probs,
    ):
        goals = np.arange(
            MAX_GOALS + 1
        )

        home = poisson.pmf(
            goals,
            expected_home,
        )

        away = poisson.pmf(
            goals,
            expected_away,
        )

        matrix = np.outer(
            home,
            away,
        )

        matrix /= matrix.sum()

        home_mask = np.tril(
            np.ones_like(
                matrix,
                dtype=bool,
            ),
            k=-1,
        )

        draw_mask = np.eye(
            len(matrix),
            dtype=bool,
        )

        away_mask = np.triu(
            np.ones_like(
                matrix,
                dtype=bool,
            ),
            k=1,
        )

        for mask, target in [
            (
                home_mask,
                result_probs[0],
            ),
            (
                draw_mask,
                result_probs[1],
            ),
            (
                away_mask,
                result_probs[2],
            ),
        ]:
            current = matrix[
                mask
            ].sum()

            matrix[mask] *= (
                target / current
            )

        matrix /= matrix.sum()

        return matrix

    def predict(
            self,
            home_team,
            away_team,
            match_date=None,
            home_rest_days=None,
            away_rest_days=None,
    ):

        if match_date is None:
            match_date = (
                pd.Timestamp.today()
                .normalize()
            )

        home_team = (
            self.normalise_team(
                home_team
            )
        )

        away_team = (
            self.normalise_team(
                away_team
            )
        )

        X = self._feature_row(
            home_team,
            away_team,
            match_date,
            home_rest_days,
            away_rest_days,
        )

        raw = (
            self.model
            .predict_proba(X)[0]
        )

        calibrated = (
            self._temperature_scale(
                raw
            )
        )

        expected_home = float(
            np.clip(
                self.home_goal_model
                .predict(X)[0],
                0.05,
                6.0,
            )
        )

        expected_away = float(
            np.clip(
                self.away_goal_model
                .predict(X)[0],
                0.05,
                6.0,
            )
        )

        score_matrix = (
            self._score_matrix(
                expected_home,
                expected_away,
                calibrated,
            )
        )

        return {
            "home_team":
                home_team,

            "away_team":
                away_team,

            "home_prob":
                calibrated[0],

            "draw_prob":
                calibrated[1],

            "away_prob":
                calibrated[2],

            "expected_home_goals":
                expected_home,

            "expected_away_goals":
                expected_away,

            "score_matrix":
                score_matrix,
        }


def main():
    predictor = (
        PremierLeaguePredictor()
    )

    print()
    print("Premier League Predictor")
    print("=" * 40)

    print(
        f"Model: "
        f"{predictor.model_name}"
    )

    print(
        f"Temperature: "
        f"{predictor.temperature:.3f}"
    )

    print()

    home = input(
        "Home team: "
    )

    away = input(
        "Away team: "
    )

    date_text = input(
        "Match date YYYY-MM-DD "
        "(blank = today): "
    ).strip()

    match_date = (
        pd.Timestamp(date_text)
        if date_text
        else pd.Timestamp.today()
    )

    prediction = predictor.predict(
        home,
        away,
        match_date=match_date,
    )

    print()
    print(
        f"{prediction['home_team']} "
        f"vs "
        f"{prediction['away_team']}"
    )

    print("-" * 40)

    print(
        f"{prediction['home_team']} win: "
        f"{prediction['home_prob']:.1%}"
    )

    print(
        f"Draw: "
        f"{prediction['draw_prob']:.1%}"
    )

    print(
        f"{prediction['away_team']} win: "
        f"{prediction['away_prob']:.1%}"
    )

    print()

    print(
        f"Expected goals: "
        f"{prediction['expected_home_goals']:.2f}"
        f" - "
        f"{prediction['expected_away_goals']:.2f}"
    )


if __name__ == "__main__":
    main()