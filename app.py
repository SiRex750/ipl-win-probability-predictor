import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

VENUES = [
    "Arun Jaitley Stadium",
    "MA Chidambaram Stadium, Chepauk",
    "M Chinnaswamy Stadium",
    "Narendra Modi Stadium, Ahmedabad",
    "Rajiv Gandhi International Stadium, Uppal",
    "Sawai Mansingh Stadium",
    "Wankhede Stadium",
    "Eden Gardens",
    "Ekana Cricket Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali",
]

MODEL_FILES = {
    "Logistic Regression": "pipe_lr.pkl",
    "XGBoost": "pipe_xgb.pkl",
}
MODEL_METRIC_KEYS = {
    "Logistic Regression": "logistic",
    "XGBoost": "xgboost",
}
METRICS_FILE = "model_metrics.json"


@st.cache_resource
def load_pipeline(model_label: str):
    model_path = MODEL_FILES[model_label]
    with open(model_path, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_metrics():
    metrics_path = Path(METRICS_FILE)
    if not metrics_path.exists():
        return {}

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def overs_to_balls(overs_completed: float) -> tuple[int, float]:
    whole_overs = int(overs_completed)
    balls_part = int(round((overs_completed - whole_overs) * 10))

    if balls_part > 5:
        raise ValueError("Overs must use cricket notation. The decimal part must be between 0 and 5.")

    total_balls = whole_overs * 6 + balls_part
    overs_fraction = whole_overs + (balls_part / 6)
    return total_balls, overs_fraction


def main():
    st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")
    st.title("IPL Win Predictor")

    available_models = [label for label, file_name in MODEL_FILES.items() if Path(file_name).exists()]
    if not available_models:
        st.error("No trained model files were found. Run train_model.py first.")
        return

    model_label = st.radio("Model", available_models, horizontal=True)
    metrics = load_metrics()

    pipe = load_pipeline(model_label)

    selected_metrics = metrics.get(MODEL_METRIC_KEYS[model_label], {})
    accuracy = selected_metrics.get("accuracy")
    roc_auc = selected_metrics.get("roc_auc")
    model_log_loss = selected_metrics.get("log_loss")

    if accuracy is not None or roc_auc is not None or model_log_loss is not None:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            if accuracy is not None:
                st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
        with metric_col2:
            if roc_auc is not None:
                st.metric("ROC-AUC", f"{roc_auc:.4f}")
        with metric_col3:
            if model_log_loss is not None:
                st.metric("Log Loss", f"{model_log_loss:.4f}")

    team_col1, team_col2 = st.columns(2)
    with team_col1:
        batting_team = st.selectbox("Batting Team", TEAMS)
    with team_col2:
        bowling_team = st.selectbox("Bowling Team", TEAMS, index=1)

    venue = st.selectbox("Venue", VENUES)

    state_col1, state_col2 = st.columns(2)
    with state_col1:
        target_score = st.number_input("Target Score", min_value=1, max_value=300, value=180, step=1)
        current_score = st.number_input("Current Score", min_value=0, max_value=300, value=75, step=1)
    with state_col2:
        overs_completed = st.number_input(
            "Overs Completed",
            min_value=0.0,
            max_value=20.0,
            value=10.2,
            step=0.1,
            format="%.1f",
        )
        wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2, step=1)

    if st.button("Predict Win Probability", type="primary", use_container_width=True):
        if batting_team == bowling_team:
            st.error("Batting team and bowling team must be different.")
            return

        try:
            balls_bowled, overs_fraction = overs_to_balls(float(overs_completed))
        except ValueError as error:
            st.error(str(error))
            return

        if balls_bowled > 120:
            st.error("Overs completed cannot exceed 20 overs.")
            return

        runs_left = max(int(target_score) - int(current_score), 0)
        balls_left = max(120 - balls_bowled, 0)
        wickets_left = max(10 - int(wickets_lost), 0)
        crr = 0.0 if overs_fraction == 0 else float(current_score) / overs_fraction
        rrr = 0.0 if balls_left == 0 else (runs_left * 6) / balls_left

        input_df = pd.DataFrame(
            [
                {
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "venue": venue,
                    "runs_left": runs_left,
                    "balls_left": balls_left,
                    "wickets_left": wickets_left,
                    "target_score": int(target_score),
                    "crr": crr,
                    "rrr": rrr,
                }
            ]
        )

        probabilities = pipe.predict_proba(input_df)[0]
        bowling_probability = float(probabilities[0])
        batting_probability = float(probabilities[1])

        st.subheader("Prediction")
        st.caption(f"Using model: {model_label}")

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.markdown(
                f"<h3 style='text-align: center;'>{batting_team}</h3>"
                f"<h1 style='text-align: center; color: #1f77b4;'>{batting_probability * 100:.2f}%</h1>",
                unsafe_allow_html=True,
            )
            st.progress(batting_probability)

        with result_col2:
            st.markdown(
                f"<h3 style='text-align: center;'>{bowling_team}</h3>"
                f"<h1 style='text-align: center; color: #d62728;'>{bowling_probability * 100:.2f}%</h1>",
                unsafe_allow_html=True,
            )
            st.progress(bowling_probability)

        st.caption(
            f"Runs left: {runs_left} | Balls left: {balls_left} | Wickets left: {wickets_left} | "
            f"CRR: {crr:.2f} | RRR: {rrr:.2f}"
        )


if __name__ == "__main__":
    main()