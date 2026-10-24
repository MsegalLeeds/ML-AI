from pathlib import Path

import pandas as pd
import streamlit as st

from predict_match import PremierLeaguePredictor


BASE_DIR = Path(__file__).resolve().parent
SIMULATION_FILE = BASE_DIR / "league_simulation.csv"


@st.cache_resource
def load_predictor():
    return PremierLeaguePredictor()


@st.cache_data
def load_simulation():
    if not SIMULATION_FILE.exists():
        return None

    return pd.read_csv(SIMULATION_FILE)


predictor = load_predictor()


st.set_page_config(
    page_title="Premier League Predictor",
    layout="wide",
)


st.title("Premier League Predictor")

st.caption(
    f"Model: {predictor.model_name}"
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

    if st.button(
        "Predict match",
        type="primary",
    ):
        prediction = predictor.predict(
            home_team,
            away_team,
        )

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

    simulation = load_simulation()

    if simulation is None:

        st.warning(
            "league_simulation.csv has not been generated yet."
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
            use_container_width=True,
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