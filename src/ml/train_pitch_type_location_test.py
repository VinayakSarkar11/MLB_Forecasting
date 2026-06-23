import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder

from src.ml.load_data import load_model1_features
from src.ml.train_model1 import time_based_split

# Predicting what the pitcher will throw next, not how the batter reacts —
# so every feature describing THIS pitch's own physics is excluded
# (pitch_type, release_speed, release_spin_rate, pfx_x/z, plate_x/z, zone,
# arm_angle, and the bat_speed_velo_gap_* terms, which are derived from
# this pitch's release_speed). Using those would mean predicting the
# target from the target. Only context known before the pitch is thrown
# is kept: count/game state, baserunners, batter's rolling tendencies, and
# the pitcher's own count-specific pitch-mix history (computed from prior
# pitches only, so it's not leakage).

CATEGORICAL_FEATURES = ["stand", "p_throws", "home_team", "if_fielding_alignment", "of_fielding_alignment"]

NUMERIC_FEATURES = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "bat_score",
    "fld_score",
    "bat_score_diff",
    "num_runners_on",
    "runner_in_scoring_position",
    "pitcher_days_since_prev_game",
    "n_thruorder_pitcher",
    "contact_rate_recent_30d",
    "swing_rate_recent_30d",
    "avg_bat_speed_recent_30d",
    "avg_swing_length_recent_30d",
    "avg_miss_distance_recent_30d",
    "contact_rate_mid_31_100d",
    "swing_rate_mid_31_100d",
    "avg_bat_speed_mid_31_100d",
    "avg_swing_length_mid_31_100d",
    "avg_miss_distance_mid_31_100d",
    "contact_rate_distant_100d_plus",
    "swing_rate_distant_100d_plus",
    "avg_bat_speed_distant_100d_plus",
    "avg_swing_length_distant_100d_plus",
    "avg_miss_distance_distant_100d_plus",
    "pitcher_fastball_rate_this_count",
    "pitcher_breaking_rate_this_count",
    "pitcher_offspeed_rate_this_count",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PITCH_CATEGORY_MAP = {
    "FF": "Fastball", "FT": "Fastball", "SI": "Fastball", "FC": "Fastball",
    "SL": "Breaking", "CU": "Breaking", "KC": "Breaking", "CS": "Breaking", "SV": "Breaking", "ST": "Breaking",
    "CH": "Offspeed", "FS": "Offspeed", "FO": "Offspeed", "SC": "Offspeed", "KN": "Offspeed", "EP": "Offspeed",
}


def prepare_features(df):
    X = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def train_and_eval(target_name, X_train, y_train, X_test, y_test, label_encoder, experiment_name):
    num_class = len(label_encoder.classes_)
    categorical_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"xgboost_{target_name}"):
        xgb_params = {
            "objective": "multi:softprob",
            "num_class": num_class,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "enable_categorical": True,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
        }
        mlflow.log_params(xgb_params)

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)

        xgb_preds = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)

        xgb_accuracy = accuracy_score(y_test, xgb_preds)
        xgb_logloss = log_loss(y_test, xgb_proba, labels=list(range(num_class)))
        mlflow.log_metric("accuracy", xgb_accuracy)
        mlflow.log_metric("log_loss", xgb_logloss)
        mlflow.xgboost.log_model(xgb_model, "model", model_format="json")

        print(f"\n=== XGBoost ({target_name}) ===")
        print(f"Accuracy: {xgb_accuracy:.4f} | Log Loss: {xgb_logloss:.4f}")
        print(classification_report(
            y_test, xgb_preds,
            labels=list(range(num_class)),
            target_names=[str(c) for c in label_encoder.classes_],
        ))

    with mlflow.start_run(run_name=f"lightgbm_{target_name}"):
        lgb_params = {
            "objective": "multiclass",
            "num_class": num_class,
            "metric": "multi_logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
        }
        mlflow.log_params(lgb_params)

        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train, categorical_feature=categorical_cols)

        lgb_preds = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)

        lgb_accuracy = accuracy_score(y_test, lgb_preds)
        lgb_logloss = log_loss(y_test, lgb_proba, labels=list(range(num_class)))
        mlflow.log_metric("accuracy", lgb_accuracy)
        mlflow.log_metric("log_loss", lgb_logloss)
        mlflow.lightgbm.log_model(lgb_model, "model")

        print(f"\n=== LightGBM ({target_name}) ===")
        print(f"Accuracy: {lgb_accuracy:.4f} | Log Loss: {lgb_logloss:.4f}")
        print(classification_report(
            y_test, lgb_preds,
            labels=list(range(num_class)),
            target_names=[str(c) for c in label_encoder.classes_],
        ))


def main():
    df = load_model1_features()
    df["pitch_category"] = df["pitch_type"].map(PITCH_CATEGORY_MAP).fillna("Other")
    train_df, test_df, cutoff_date = time_based_split(df)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows | cutoff: {cutoff_date.date()}")

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    pitch_type_encoder = LabelEncoder()
    y_train_type = pitch_type_encoder.fit_transform(train_df["pitch_category"])
    y_test_type = pitch_type_encoder.transform(test_df["pitch_category"])
    train_and_eval(
        "pitch_category", X_train, y_train_type, X_test, y_test_type,
        pitch_type_encoder, "pitch_type_location_test",
    )

    zone_encoder = LabelEncoder()
    y_train_zone = zone_encoder.fit_transform(train_df["zone"].astype(str))
    y_test_zone = zone_encoder.transform(test_df["zone"].astype(str))
    train_and_eval(
        "zone", X_train, y_train_zone, X_test, y_test_zone,
        zone_encoder, "pitch_type_location_test",
    )


if __name__ == "__main__":
    main()
