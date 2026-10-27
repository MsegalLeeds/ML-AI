from pathlib import Path

import pandas as pd
import streamlit as st

from predict_match import MODEL_FILE, PremierLeaguePredictor
from prepare_data import DATA_DIR
from league_simulator import FIXTURE_FILE


BASE_DIR = Path(__file__).resolve().parent
SIMULATION_FILE = BASE_DIR / "league_simulation.csv"

st.set_page_config(page_title="Premier League Predictor", layout="wide")


def file_version(path):
    if not path.exists():
        return (str(path), None, None)
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


@st.cache_resource
def load_predictor(versions):
    # File versions are cache keys so retraining and CSV updates take effect.
    return PremierLeaguePredictor()


@st.cache_data
def load_simulation(version):
    if not SIMULATION_FILE.exists():
        return None

    return pd.read_csv(SIMULATION_FILE)


try:
    predictor = load_predictor(tuple(
        file_version(path) for path in [MODEL_FILE, *sorted(DATA_DIR.glob("*.csv"))]
    ))
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()


st.title("Premier League Predictor")

st.caption(
    f"Model: {predictor.model_name} | "
    f"Trained through: {predictor.trained_through.date()} | "
    f"Results through: {predictor.matches['Date'].max().date()}"
)


tab_match, tab_league = st.tabs([
    "Match Predictor",
    "League Predictor",
])


# =========================================================
# MATCH PREDICTOR
# =========================================================

with tab_match:

    st.header("Match Predictor")

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox(
            "Home team",
            predictor.teams,
            index=0,
        )

    with col2:
        away_options = [
            team
            for team in predictor.teams
            if team != home_team
        ]

        away_team = st.selectbox(
            "Away team",
            away_options,
            index=0,
        )

    first_prediction_date = max(
        predictor.matches["Date"].max().normalize(), predictor.trained_through
    ) + pd.Timedelta(days=1)
    match_date = st.date_input(
        "Match date",
        value=max(pd.Timestamp.today().normalize(), first_prediction_date).date(),
        min_value=first_prediction_date.date(),
        help="Rest days use this date and the latest completed Premier League results.",
    )

    if st.button(
        "Predict match",
        type="primary",
    ):
        try:
            prediction = predictor.predict(home_team, away_team, match_date=match_date)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.subheader(
            f"{prediction['home_team']} vs "
            f"{prediction['away_team']}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                f"{home_team} win",
                f"{prediction['home_prob']:.1%}",
            )

        with c2:
            st.metric(
                "Draw",
                f"{prediction['draw_prob']:.1%}",
            )

        with c3:
            st.metric(
                f"{away_team} win",
                f"{prediction['away_prob']:.1%}",
            )

        st.subheader("Expected goals")

        g1, g2 = st.columns(2)

        with g1:
            st.metric(
                home_team,
                f"{prediction['expected_home_goals']:.2f}",
            )

        with g2:
            st.metric(
                away_team,
                f"{prediction['expected_away_goals']:.2f}",
            )


# =========================================================
# LEAGUE SIMULATION
# =========================================================

with tab_league:

    st.header("League Predictor")

    simulation_version = file_version(SIMULATION_FILE)
    simulation = load_simulation(simulation_version)
    inputs = [MODEL_FILE, FIXTURE_FILE, *DATA_DIR.glob("*.csv")]
    stale = simulation is not None and any(
        path.exists() and path.stat().st_mtime_ns > simulation_version[1]
        for path in inputs
    )

    if simulation is None or stale:

        st.warning(
            "league_simulation.csv is missing or older than its model, results, "
            "or fixture schedule. Regenerate it to see current league forecasts."
        )

        st.code(
            "python league_simulator.py"
        )

    else:

        display = simulation.copy()

        for column in [
            "Title",
            "Top4",
            "Relegation",
        ]:
            display[column] = (
                display[column] * 100
            ).round(1)

        display = display.rename(
            columns={
                "Title": "Title %",
                "Top4": "Top 4 %",
                "Relegation": "Relegation %",
                "AvgFinish": "Average Finish",
                "CurrentPts": "Current Points",
            }
        )

        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Title probabilities"
        )

        title_chart = (
            simulation[
                ["Team", "Title"]
            ]
            .set_index("Team")
            .sort_values(
                "Title",
                ascending=False,
            )
        )

        st.bar_chart(
            title_chart
        )

        st.subheader(
            "Top 4 probabilities"
        )

        top4_chart = (
            simulation[
                ["Team", "Top4"]
            ]
            .set_index("Team")
            .sort_values(
                "Top4",
                ascending=False,
            )
        )

        st.bar_chart(
            top4_chart
        )

        st.subheader(
            "Relegation probabilities"
        )

        relegation_chart = (
            simulation[
                ["Team", "Relegation"]
            ]
            .set_index("Team")
            .sort_values(
                "Relegation",
                ascending=False,
            )
        )

        st.bar_chart(
            relegation_chart
        )
