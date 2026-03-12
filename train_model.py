import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def build_preprocessor(categorical_columns):
    return ColumnTransformer(
        transformers=[
            (
                "team_venue_ohe",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        ],
        remainder="passthrough",
    )


def main():
    df = pd.read_csv("ipl_model_ready.csv")
    df = df.replace([np.inf, -np.inf], 0)

    feature_columns = [
        "batting_team",
        "bowling_team",
        "venue",
        "runs_left",
        "balls_left",
        "wickets_left",
        "target_score",
        "crr",
        "rrr",
    ]
    categorical_columns = ["batting_team", "bowling_team", "venue"]
    groups = df["match_id"]

    X = df[feature_columns]
    y = df["result"]

    match_ids = groups.drop_duplicates().to_numpy()
    train_match_ids, test_match_ids = train_test_split(
        match_ids,
        test_size=0.2,
        random_state=42,
    )

    train_full_mask = groups.isin(train_match_ids)
    test_mask = groups.isin(test_match_ids)

    X_train_full = X.loc[train_full_mask]
    y_train_full = y.loc[train_full_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    baseline_preprocessor = build_preprocessor(categorical_columns)

    baseline_pipe = Pipeline(
        steps=[
            ("preprocessor", baseline_preprocessor),
            ("model", LogisticRegression(solver="liblinear", max_iter=1000)),
        ]
    )

    baseline_pipe.fit(X_train_full, y_train_full)
    baseline_predictions = baseline_pipe.predict(X_test)
    baseline_probabilities = baseline_pipe.predict_proba(X_test)
    baseline_positive_probabilities = baseline_probabilities[:, 1]
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    baseline_roc_auc = roc_auc_score(y_test, baseline_positive_probabilities)
    baseline_log_loss = log_loss(y_test, baseline_probabilities)

    train_match_ids, val_match_ids = train_test_split(
        train_match_ids,
        test_size=0.1,
        random_state=42,
    )

    train_mask = groups.isin(train_match_ids)
    val_mask = groups.isin(val_match_ids)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    preprocessor = build_preprocessor(categorical_columns)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=8.0,
        gamma=0.5,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=10,
        tree_method="hist",
    )

    model.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_val_processed, y_val)],
        verbose=False,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    y_pred = pipe.predict(X_test)
    xgb_probabilities = pipe.predict_proba(X_test)
    xgb_positive_probabilities = xgb_probabilities[:, 1]
    xgb_accuracy = accuracy_score(y_test, y_pred)
    xgb_roc_auc = roc_auc_score(y_test, xgb_positive_probabilities)
    xgb_log_loss = log_loss(y_test, xgb_probabilities)
    sample_row = X_test.sample(n=1, random_state=42)
    sample_probability = pipe.predict_proba(sample_row)[0]

    print(f"Logistic Regression accuracy: {baseline_accuracy:.4f}")
    print(f"Logistic Regression ROC-AUC: {baseline_roc_auc:.4f}")
    print(f"Logistic Regression log loss: {baseline_log_loss:.4f}")
    print(f"XGBoost accuracy: {xgb_accuracy:.4f}")
    print(f"XGBoost ROC-AUC: {xgb_roc_auc:.4f}")
    print(f"XGBoost log loss: {xgb_log_loss:.4f}")
    print(f"Accuracy change: {xgb_accuracy - baseline_accuracy:+.4f}")
    print(f"Train matches: {len(train_match_ids)} | Validation matches: {len(val_match_ids)} | Test matches: {len(test_match_ids)}")
    print("Probability output for a sample row:", sample_probability.tolist())

    metrics = {
        "logistic": {
            "accuracy": round(float(baseline_accuracy), 4),
            "roc_auc": round(float(baseline_roc_auc), 4),
            "log_loss": round(float(baseline_log_loss), 4),
        },
        "xgboost": {
            "accuracy": round(float(xgb_accuracy), 4),
            "roc_auc": round(float(xgb_roc_auc), 4),
            "log_loss": round(float(xgb_log_loss), 4),
        },
    }

    with open("pipe_lr.pkl", "wb") as file:
        pickle.dump(baseline_pipe, file)

    with open("pipe_xgb.pkl", "wb") as file:
        pickle.dump(pipe, file)

    with open("pipe.pkl", "wb") as file:
        pickle.dump(pipe, file)

    with open("model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Saved logistic pipeline to pipe_lr.pkl")
    print("Saved XGBoost pipeline to pipe_xgb.pkl")
    print("Saved default pipeline to pipe.pkl")
    print("Saved model metrics to model_metrics.json")


if __name__ == "__main__":
    main()