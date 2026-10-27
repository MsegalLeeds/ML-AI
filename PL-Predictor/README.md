# Premier League predictor

Predict home-win, draw and away-win probabilities, estimate goals, and simulate the remaining league season. A Streamlit app displays match predictions and league finishing probabilities.

## Setup and run

Use Python 3.12 or 3.13. Run these commands from the repository root after cloning:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r PL-Predictor/requirements.txt
python -m unittest discover -s PL-Predictor -v
python PL-Predictor/prepare_data.py
python PL-Predictor/train_model.py
python PL-Predictor/league_simulator.py
python -m streamlit run PL-Predictor/app.py
```

On Windows use `.venv\Scripts\activate`. The dependency file pins the tested direct dependencies, including SciPy and Streamlit. Transitive dependencies are resolved by pip. Re-train artifacts when changing the Python or scikit-learn version.

For an interactive terminal prediction:

```bash
python PL-Predictor/predict_match.py
```

Team aliases accept case differences and surrounding spaces, for example `man utd` or `Spurs`. The match date must be later than both the latest completed result and the model's training cutoff. This prevents a historical fixture from using later results. Predictions require five completed matches in the current season for each team, with complete form statistics.

## Data and generated files

- `data/SeasonYYYY-YYYY.csv` contains completed results in Football-Data column format. Required columns are `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HS`, `AS`, `HST`, `AST`. Dates use day/month/year. `FTR` is `H`, `D` or `A`.
- Optional `B365H`, `B365D`, `B365A` columns support bookmaker comparisons.
- `fixtures/epl-2026.csv` contains the full 2026/27 schedule, with `Date`, `Home Team` and `Away Team` columns. Dates use day/month/year, with optional kickoff times. Update `FIXTURE_FILE` in `league_simulator.py` when moving to another season.
- Each ordered home/away pairing must appear once. Duplicate fixtures, missing pairings, invalid dates and multiple games for one team on a date are rejected. Completed games are removed using current-season home/away pairings. An unplayed game on or before the results cutoff requires corrected results or a rescheduled date.

To update results, replace or extend the season CSV, update postponed fixture dates, then rerun preparation, training and simulation in that order. The app reloads changed input files on its next rerun and asks for regeneration when the simulation file is older than the model, result files or fixture schedule. Freshness detection uses local file modification times; copied files with preserved timestamps need manual regeneration.

Generated files stay local and are ignored by Git:

| File | Created by | Purpose |
| --- | --- | --- |
| `model_data.csv` | `prepare_data.py` | Pre-match features and targets |
| `premier_league_model.joblib` | `train_model.py` | Fitted models, feature order, temperature and training cutoff |
| `training_metrics.json` | `train_model.py` | Test metrics, baseline, calibration and sample counts |
| `league_simulation.csv` | `league_simulator.py` | Title, top-four, relegation and average-finish estimates |

Only load model artifacts you created or trust; joblib files can execute Python code when loaded.

## What the model does

1. Build current-season rolling form from the previous five matches. Elo carries across seasons. Training features exclude the current result. Rest is measured in calendar days since the previous Premier League game and capped at 14.
2. Train scaled logistic regression on seasons before 2024/25 and tune a probability temperature on 2024/25.
3. Refit on seasons before 2025/26 and evaluate on usable 2025/26 fixtures. The temperature stays fixed for this test.
4. Refit production logistic and Poisson goal models on all usable results, including the current season. The saved production artifact is for future predictions, not historical evaluation.
5. Build a truncated 0–10 goal score matrix from the Poisson models and rescale its home-win/draw/away-win regions to match the calibrated logistic probabilities.

The displayed expected goals are the Poisson estimates before this rescaling. They need not equal the means of the adjusted score matrix.

`training_metrics.json` compares raw and calibrated logistic predictions with a constant baseline derived from historical outcome frequencies, all on the same test fixtures. Lower log loss and Brier score are better. Accuracy alone does not measure probability quality. The validation loss is a calibration fitting score, not an independent test score.

The checked-in dataset currently has 3,490 completed matches through 19 October 2026. With the pinned dependencies, 330 usable 2025/26 test fixtures give:

| Predictor | Accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: |
| Historical-frequency baseline | 42.1% | 1.0869 | 0.6578 |
| Raw logistic | 48.8% | 1.0324 | 0.6195 |
| Calibrated logistic | 48.8% | 1.0260 | 0.6154 |

These results describe the supplied files. Data provenance and any model selection performed before this repository review have not been independently verified.

## League simulation assumptions

The simulator samples 20,000 seasons using seed 42. Completed points and goals are retained. Remaining games use their scheduled dates; rest days advance after each scheduled game. Form and Elo remain fixed at the latest observed values rather than changing with simulated results.

Rankings use points, goal difference and goals scored, then a random tie-break. They do not implement every official head-to-head or playoff rule. Top four is a finishing-position probability, not a guarantee of qualification under all competition rules. Cup and European matches do not contribute to rest days.

## Benchmarks and checks

```bash
python PL-Predictor/benchmark.py
python PL-Predictor/calibration_benchmark.py
python PL-Predictor/poisson_benchmark.py
python PL-Predictor/ensemble_benchmark.py
```

`benchmark.py` compares the base and venue-feature models with Bet365 on one common set of fixtures. These experimental benchmark scripts do not select or replace the production model. `xgboost_benchark.py` retains its existing filename and needs the optional `xgboost` package.

```bash
python -m unittest discover -s PL-Predictor -v
python -m compileall -q PL-Predictor
python -m pip check
```

GitHub Actions runs these checks on Python 3.12 and 3.13. Tests cover pre-match feature leakage, rest-day caps, prediction inputs, fixture completeness, score probabilities, missing-model guidance, and an end-to-end training/simulation/app flow. Integration tests use temporary generated files so they do not replace your saved model.
