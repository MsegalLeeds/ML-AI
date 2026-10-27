import unittest

import pandas as pd

from prepare_data import (
    add_elo_ratings,
    add_rest_days,
    add_rolling_features,
    add_venue_features,
    build_team_history,
    build_model_dataset,
    calculate_current_elo_ratings,
    get_venue_feature_columns,
)


class PipelineTests(unittest.TestCase):
    def test_current_result_cannot_change_its_prediction_features(self):
        def build(matches):
            matches = add_elo_ratings(add_rest_days(matches))
            history = add_venue_features(add_rolling_features(build_team_history(matches)))
            return build_model_dataset(matches, history)

        matches = self.make_matches()
        original = build(matches)
        matches.loc[7, ['FTHG', 'FTAG', 'FTR', 'HS', 'HST']] = [9, 0, 'H', 30, 20]
        changed = build(matches)
        pd.testing.assert_series_equal(
            original.iloc[-1][get_venue_feature_columns()],
            changed.iloc[-1][get_venue_feature_columns()],
        )

    def test_rest_days_follow_previous_matches_and_cap_breaks(self):
        matches = self.make_matches().iloc[:3].copy()
        matches['Date'] = pd.to_datetime(['2025-08-01', '2025-08-05', '2025-09-01'])
        result = add_rest_days(matches)
        self.assertTrue(pd.isna(result.iloc[0].HomeRestDays))
        self.assertEqual(result.iloc[1].HomeRestDays, 4)
        self.assertEqual(result.iloc[1].AwayRestDays, 4)
        self.assertEqual(result.iloc[2].HomeRestDays, 14)

    def make_matches(self) -> pd.DataFrame:
        rows = []

        # Eight alternating fixtures create enough overall and
        # venue-specific history for both teams.
        for i in range(8):
            if i % 2 == 0:
                home, away = "Alpha", "Beta"
                fthg, ftag = 2, 0
                hs, ass = 12, 8
                hst, ast = 6, 3
                ftr = "H"
            else:
                home, away = "Beta", "Alpha"
                fthg, ftag = 1, 1
                hs, ass = 10, 9
                hst, ast = 4, 4
                ftr = "D"

            rows.append({
                "Date": pd.Timestamp("2025-08-01")
                + pd.Timedelta(days=i * 7),
                "Season": "Season2025-2026",
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": fthg,
                "FTAG": ftag,
                "FTR": ftr,
                "HS": hs,
                "AS": ass,
                "HST": hst,
                "AST": ast,
                "B365H": 2.0,
                "B365D": 3.2,
                "B365A": 3.8,
            })

        return pd.DataFrame(rows)

    def test_first_elo_is_pre_match(self):
        matches = self.make_matches()
        rated = add_elo_ratings(matches)

        self.assertEqual(
            rated.iloc[0]["HomeElo"],
            1500.0,
        )
        self.assertEqual(
            rated.iloc[0]["AwayElo"],
            1500.0,
        )

    def test_current_elo_changes_after_results(self):
        matches = self.make_matches()
        ratings = calculate_current_elo_ratings(
            matches
        )

        self.assertIn("Alpha", ratings)
        self.assertIn("Beta", ratings)
        self.assertNotEqual(
            ratings["Alpha"],
            1500.0,
        )

    def test_rolling_features_exclude_current_match(self):
        matches = self.make_matches()
        history = build_team_history(add_elo_ratings(add_rest_days(matches)))
        history = add_rolling_features(
            history,
            window=5,
        )

        alpha = history[
            history["Team"] == "Alpha"
        ].sort_values("Date")

        sixth = alpha.iloc[5]

        # Alpha's first five points are 3,1,3,1,3 => 2.2 average.
        self.assertAlmostEqual(
            sixth["Points_last5"],
            2.2,
            places=6,
        )

    def test_venue_features_exist_after_three_prior_games(self):
        matches = self.make_matches()
        history = build_team_history(add_elo_ratings(add_rest_days(matches)))
        history = add_venue_features(
            history,
            window=3,
        )

        alpha_home = history[
            (history["Team"] == "Alpha")
            & (history["Venue"] == "Home")
        ].sort_values("Date")

        # The fourth Alpha home match has three prior home matches.
        self.assertFalse(
            pd.isna(
                alpha_home.iloc[3][
                    "Points_venue_last3"
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
