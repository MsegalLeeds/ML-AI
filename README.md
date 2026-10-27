# ML-AI

Machine-learning experiments, starting with a Premier League match predictor and season simulator.

## Projects

| Project | Status |
| --- | --- |
| [Premier League predictor](PL-Predictor/README.md) | Working training pipeline, match predictions, league simulation and Streamlit app |
| Stock analysis | Idea only; no implementation yet |
| Drone project | Idea only; requirements not defined yet |

## Quick start

Use Python 3.12 or 3.13. From the repository root:

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

On Windows, activate with `.venv\Scripts\activate` instead. Python 3.12+ is required because existing scripts use multiline f-string expressions.

Read the [predictor guide](PL-Predictor/README.md) for data updates, model assumptions and evaluation. The [assessment](ASSESSMENT.md) records the reliability fixes and remaining priorities.
