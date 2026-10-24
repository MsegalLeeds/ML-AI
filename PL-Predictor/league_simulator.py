from pathlib import Path

import numpy as np
import pandas as pd

from predict_match import (
    PremierLeaguePredictor,
)


BASE_DIR = Path(__file__).resolve().parent

FIXTURE_FILE = (
    BASE_DIR
    / "fixtures"
    / "epl-2026.csv"
)

SIMULATIONS = 20_000

RANDOM_SEED = 42


def current_table(
    predictor,
):
    matches = predictor.matches[
        predictor.matches["Season"]
        == predictor.season
    ]

    teams = sorted(
        set(matches["HomeTeam"])
        | set(matches["AwayTeam"])
    )

    table = {
        team: {
            "P": 0,
            "W": 0,
            "D": 0,
            "L": 0,
            "GF": 0,
            "GA": 0,
            "Pts": 0,
        }
        for team in teams
    }

    for _, match in matches.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        hg = int(match["FTHG"])
        ag = int(match["FTAG"])

        table[home]["P"] += 1
        table[away]["P"] += 1

        table[home]["GF"] += hg
        table[home]["GA"] += ag

        table[away]["GF"] += ag
        table[away]["GA"] += hg

        if hg > ag:
            table[home]["W"] += 1
            table[away]["L"] += 1
            table[home]["Pts"] += 3

        elif hg < ag:
            table[away]["W"] += 1
            table[home]["L"] += 1
            table[away]["Pts"] += 3

        else:
            table[home]["D"] += 1
            table[away]["D"] += 1

            table[home]["Pts"] += 1
            table[away]["Pts"] += 1

    return table


def load_remaining_fixtures(
    predictor,
):
    fixtures = pd.read_csv(
        FIXTURE_FILE
    )

    fixtures["Date"] = pd.to_datetime(
        fixtures["Date"],
        dayfirst=True,
        errors="coerce",
    )

    fixtures = fixtures.rename(
        columns={
            "Home Team": "HomeTeam",
            "Away Team": "AwayTeam",
        }
    )

    fixtures["HomeTeam"] = (
        fixtures["HomeTeam"]
        .apply(
            predictor.normalise_team
        )
    )

    fixtures["AwayTeam"] = (
        fixtures["AwayTeam"]
        .apply(
            predictor.normalise_team
        )
    )

    current_matches = (
        predictor.matches[
            predictor.matches["Season"]
            == predictor.season
        ]
    )

    played = set(
        zip(
            current_matches["HomeTeam"],
            current_matches["AwayTeam"],
        )
    )

    fixtures["played"] = [
        (
            home,
            away,
        ) in played

        for home, away
        in zip(
            fixtures["HomeTeam"],
            fixtures["AwayTeam"],
        )
    ]

    remaining = fixtures[
        ~fixtures["played"]
    ].copy()

    return remaining.sort_values(
        "Date"
    )


def precompute_predictions(
    predictor,
    fixtures,
):
    predictions = []

    print()
    print(
        f"Predicting "
        f"{len(fixtures)} "
        f"remaining fixtures..."
    )

    for _, fixture in fixtures.iterrows():

        prediction = (
            predictor.predict(
                fixture["HomeTeam"],
                fixture["AwayTeam"],
            )
        )

        matrix = (
            prediction[
                "score_matrix"
            ]
        )

        cdf = (
            matrix
            .ravel()
            .cumsum()
        )

        cdf[-1] = 1.0

        predictions.append({
            "home":
                prediction["home_team"],

            "away":
                prediction["away_team"],

            "cdf":
                cdf,

            "matrix_size":
                matrix.shape[0],
        })

    return predictions


def simulate():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    predictor = (
        PremierLeaguePredictor()
    )

    table = current_table(
        predictor
    )

    fixtures = (
        load_remaining_fixtures(
            predictor
        )
    )

    fixture_predictions = (
        precompute_predictions(
            predictor,
            fixtures,
        )
    )

    teams = sorted(
        table.keys()
    )

    team_index = {
        team: index
        for index, team
        in enumerate(teams)
    }

    n_teams = len(teams)

    current_pts = np.array([
        table[t]["Pts"]
        for t in teams
    ])

    current_gf = np.array([
        table[t]["GF"]
        for t in teams
    ])

    current_ga = np.array([
        table[t]["GA"]
        for t in teams
    ])

    points = np.tile(
        current_pts,
        (
            SIMULATIONS,
            1,
        ),
    )

    gf = np.tile(
        current_gf,
        (
            SIMULATIONS,
            1,
        ),
    )

    ga = np.tile(
        current_ga,
        (
            SIMULATIONS,
            1,
        ),
    )

    for fixture in fixture_predictions:

        home_idx = team_index[
            fixture["home"]
        ]

        away_idx = team_index[
            fixture["away"]
        ]

        random_values = rng.random(
            SIMULATIONS
        )

        score_indices = (
            np.searchsorted(
                fixture["cdf"],
                random_values,
            )
        )

        size = fixture[
            "matrix_size"
        ]

        home_goals = (
            score_indices // size
        )

        away_goals = (
            score_indices % size
        )

        gf[:, home_idx] += (
            home_goals
        )

        ga[:, home_idx] += (
            away_goals
        )

        gf[:, away_idx] += (
            away_goals
        )

        ga[:, away_idx] += (
            home_goals
        )

        home_win = (
            home_goals
            > away_goals
        )

        away_win = (
            away_goals
            > home_goals
        )

        draw = (
            home_goals
            == away_goals
        )

        points[
            home_win,
            home_idx,
        ] += 3

        points[
            away_win,
            away_idx,
        ] += 3

        points[
            draw,
            home_idx,
        ] += 1

        points[
            draw,
            away_idx,
        ] += 1

    goal_difference = (
        gf - ga
    )

    # Tiny random value only resolves extremely rare
    # complete Pts/GD/GF ties.
    tie_breaker = rng.random(
        (
            SIMULATIONS,
            n_teams,
        )
    ) * 1e-6

    rankings = np.lexsort(
        (
            tie_breaker,
            -gf,
            -goal_difference,
            -points,
        ),
        axis=1,
    )

    finish_counts = np.zeros(
        (
            n_teams,
            n_teams,
        ),
        dtype=int,
    )

    for position in range(
        n_teams
    ):
        finish_counts[
            :,
            position,
        ] = np.bincount(
            rankings[:, position],
            minlength=n_teams,
        )

    finish_prob = (
        finish_counts
        / SIMULATIONS
    )

    results = []

    for index, team in enumerate(
        teams
    ):
        avg_finish = np.sum(
            finish_prob[index]
            * np.arange(
                1,
                n_teams + 1,
            )
        )

        results.append({
            "Team": team,

            "CurrentPts":
                table[team]["Pts"],

            "Title":
                finish_prob[
                    index,
                    0,
                ],

            "Top4":
                finish_prob[
                    index,
                    :4,
                ].sum(),

            "Relegation":
                finish_prob[
                    index,
                    -3:,
                ].sum(),

            "AvgFinish":
                avg_finish,
        })

    results = pd.DataFrame(
        results
    )

    results = results.sort_values(
        [
            "Title",
            "Top4",
            "AvgFinish",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    print()
    print(
        f"Simulations: "
        f"{SIMULATIONS:,}"
    )

    print(
        f"Remaining fixtures: "
        f"{len(fixtures)}"
    )

    print()

    print(
        results.to_string(
            index=False,
            formatters={
                "Title":
                    "{:.1%}".format,

                "Top4":
                    "{:.1%}".format,

                "Relegation":
                    "{:.1%}".format,

                "AvgFinish":
                    "{:.2f}".format,
            },
        )
    )

    output = (
        BASE_DIR
        / "league_simulation.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print()
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    simulate()