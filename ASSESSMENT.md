# Repository assessment

Reviewed the repository at `9c0545f` on 27 October 2026. The Premier League predictor is the only implemented project.

## Fixed in this change

| Finding | Change and evidence |
| --- | --- |
| Two of four pipeline tests failed because fixtures lacked Elo/rest columns | Tests now follow the actual preparation pipeline. Added a test that changes a result and confirms its own prediction features stay unchanged. |
| Simulator ignored fixture dates and reused today's rest for every remaining game | Predictions now run in schedule order, using calendar-day rest since the preceding completed or scheduled game. |
| Duplicate or incomplete fixture lists silently changed the simulated season | Full home/away schedule validation rejects missing pairings, duplicates, invalid dates and same-day conflicts. |
| Historical match dates could use future results and produce negative rest | Predict rejects dates at or before the training/results cutoff, same-team fixtures and invalid rest overrides. Overrides use the same 14-day cap as training. |
| Prediction means silently skipped missing recent statistics | Incomplete five-match windows now produce a clear error instead of using a different window from training. |
| Web app setup omitted dependencies and failed with a traceback without a model | Added SciPy and Streamlit, pinned tested direct versions, documented Python 3.12+, and displayed actionable setup guidance. |
| Cached models/results remained stale after file changes | Cache keys now include input file versions. Older league simulation files are hidden with regeneration instructions. |
| Model evaluation existed only in terminal output | Training writes sample counts, raw/calibrated metrics and a historical-frequency baseline to `training_metrics.json`. |
| README described an obsolete production-model selection process | Replaced it with current training, evaluation, simulation and update instructions. |
| No automated checks | Added regression tests and a GitHub Actions workflow, including a temporary-file end-to-end run. |

## Verification

- Prepared all 3,490 supplied completed matches; 2,987 have complete base-model features.
- Trained and evaluated the production approach on 330 usable 2025/26 test fixtures: calibrated log loss 1.0260, Brier 0.6154, accuracy 48.8%. Historical-frequency baseline log loss was 1.0869 on the same fixtures.
- Ran all 20,000 season simulations for the 310 remaining fixtures.
- Verified a match prediction and the league table in a live browser. Streamlit's chart dependencies emit non-blocking deprecation/scale-binding warnings; no browser errors were observed.
- Added automated tests for stale league output and successful match prediction through Streamlit.

## Next priorities

1. **Validate predictive quality across multiple seasons.** Add expanding-window backtests and calibration plots. Report performance by season and compare against market and frequency baselines on identical fixtures. Preserve an untouched future period before choosing additional models or features.
2. **Track data provenance and validation.** Record source URLs, retrieval dates and checksums. The current loader drops rows with invalid dates/results and does not comprehensively validate scores or duplicate results. Replace silent exclusions with a data-quality report before automating downloads.
3. **Measure early-season coverage.** Predictions need five complete current-season games; add and backtest an explicit fallback for promoted teams and the start of a season before enabling those forecasts.
4. **Test richer simulation assumptions.** Compare fixed-strength forecasts with simulated Elo/form updates. Include cup/European fixtures if rest is intended to represent actual workload. Document the cost and accuracy benefit before adding complexity.
5. **Define the stock and drone projects.** Choose their users, inputs, measurable outputs and evaluation datasets before adding implementations. They currently have no code or specification.

This pass addresses reproducibility and correctness. It does not establish a new predictive-accuracy gain or validate the external accuracy of the supplied datasets.
