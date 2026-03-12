# IPL Win Predictor

An IPL match win probability predictor built from ball-by-ball second-innings data.

The project preprocesses historical IPL deliveries, engineers match-state features, trains two machine learning pipelines, and serves predictions through a Streamlit web app.

## What This Project Does

- Preprocesses raw IPL ball-by-ball data from `archive/ipl_data.csv`
- Builds second-innings training data for win probability prediction
- Engineers live match-state features such as `runs_left`, `balls_left`, `crr`, and `rrr`
- Trains two models:
  - Logistic Regression
  - XGBoost
- Evaluates models using:
  - Accuracy
  - ROC-AUC
  - Log Loss
- Serves predictions in a Streamlit app with model switching

## Project Structure

- `preprocess.py` - builds `processed_deliveries.csv` from the raw dataset
- `feature_engineering.py` - creates `ipl_model_ready.csv`
- `train_model.py` - trains Logistic Regression and XGBoost pipelines
- `app.py` - Streamlit interface for live win probability prediction
- `archive/ipl_data.csv` - raw source dataset placed locally before preprocessing

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Place the raw IPL CSV at `archive/ipl_data.csv` before running the preprocessing script.

## Training Pipeline

Run the scripts in this order:

```bash
python3 preprocess.py
python3 feature_engineering.py
python3 train_model.py
```

This will generate:

- `processed_deliveries.csv`
- `ipl_model_ready.csv`
- `pipe_lr.pkl`
- `pipe_xgb.pkl`
- `pipe.pkl`
- `model_metrics.json`

## Run The App

```bash
python3 -m streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Model Inputs

Both trained pipelines expect a pandas DataFrame with these columns:

- `batting_team`
- `bowling_team`
- `venue`
- `runs_left`
- `balls_left`
- `wickets_left`
- `target_score`
- `crr`
- `rrr`

## Evaluation

The training script uses match-level splitting to avoid leakage between deliveries from the same match.

Current metrics are saved in `model_metrics.json` and shown inside the Streamlit app.