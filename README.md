# IPL Win Predictor

An IPL match win probability predictor built from ball-by-ball second-innings data.

The project preprocesses historical IPL deliveries, engineers match-state features, trains two machine learning pipelines, and serves predictions through a Streamlit web app.

## Overview

This project predicts the probability of the chasing team winning an IPL match at any point in the second innings.

It uses live match context such as score, overs completed, wickets lost, target, current run rate, and required run rate to estimate win probability through trained machine learning pipelines.

The application includes:

- a complete preprocessing and feature-engineering workflow
- two trained models for comparison
- leakage-free evaluation using match-level splits
- a Streamlit UI for interactive predictions

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

## Features

- Predicts live win probability during a chase
- Lets the user switch between Logistic Regression and XGBoost
- Displays model evaluation metrics directly in the UI
- Handles unseen venues safely with `OneHotEncoder(handle_unknown="ignore")`
- Uses match-level train, validation, and test splits to avoid leakage

## Project Structure

- `preprocess.py` - builds `processed_deliveries.csv` from the raw dataset
- `feature_engineering.py` - creates `ipl_model_ready.csv`
- `train_model.py` - trains Logistic Regression and XGBoost pipelines
- `app.py` - Streamlit interface for live win probability prediction
- `archive/ipl_data.csv` - raw source dataset placed locally before preprocessing

## Model Pipelines

The repository trains and compares two models:

- Logistic Regression with one-hot encoded categorical features
- XGBoost with tuned hyperparameters and early stopping

Saved artifacts:

- `pipe_lr.pkl` - Logistic Regression pipeline
- `pipe_xgb.pkl` - XGBoost pipeline
- `pipe.pkl` - default exported pipeline
- `model_metrics.json` - evaluation metrics for both models

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

## GitHub Pages

This repository also includes a static GitHub Pages site under `docs/` for project presentation.

Important limitation:

- GitHub Pages cannot run the Streamlit app itself because Pages only serves static files.
- The interactive predictor still needs a Python hosting platform such as Streamlit Community Cloud, Render, or Railway.

## Input Features Used For Prediction

The app constructs a single-row pandas DataFrame with these columns before sending it to the selected model pipeline:

- `batting_team`
- `bowling_team`
- `venue`
- `runs_left`
- `balls_left`
- `wickets_left`
- `target_score`
- `crr`
- `rrr`

## Current Evaluation Metrics

Latest leakage-free test metrics:

| Model | Accuracy | ROC-AUC | Log Loss |
|---|---:|---:|---:|
| Logistic Regression | 0.7799 | 0.8650 | 0.4777 |
| XGBoost | 0.7954 | 0.8736 | 0.4514 |

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

## Tech Stack

- Python
- pandas
- scikit-learn
- XGBoost
- Streamlit

## Notes

- The raw source dataset is intentionally not committed to git.
- Generated CSVs and trained model artifacts are excluded through `.gitignore`.
- The repository contains code, setup, and documentation needed to reproduce the workflow locally.