import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder

import pandas as pd

from src.ml.load_data import load_model1_features

CATEGORICAL_FEATURES = [
    "pitch_type",
    "stand",
    "p_throws",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "prev_pitch_category",
    "prev_pitch_result",
]

NUMERIC_FEATURES = [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "arm_angle",
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
    "contact_rate_career",
    "swing_rate_career",
    "avg_bat_speed_career",
    "avg_swing_length_career",
    "avg_miss_distance_career",
    "bat_speed_velo_gap_career",
    "pitcher_fastball_rate_this_count",
    "pitcher_breaking_rate_this_count",
    "pitcher_offspeed_rate_this_count",
    # batter count-specific swing rate
    "batter_swing_rate_this_count",
    # pitch sequencing
    "prev_plate_x",
    "prev_plate_z",
    "prev_release_speed",
    "pitches_in_ab",
    "fouls_2strike_in_ab",
    "fastballs_in_ab",
    "breaking_in_ab",
    "offspeed_in_ab",
    # pitcher career average plate location by pitch type (pre-pitch location prior)
    "avg_plate_x_this_pitch_career",
    "avg_plate_z_this_pitch_career",
    "avg_in_zone_rate_this_pitch_career",
    # pitcher career avg plate location by pitch type AND count
    "avg_plate_x_this_pitch_this_count",
    "avg_plate_z_this_pitch_this_count",
    "avg_in_zone_rate_this_pitch_this_count",
    # batter career miss distance by pitch type and zone context
    "avg_miss_distance_fastball_career",
    "avg_miss_distance_breaking_career",
    "avg_miss_distance_offspeed_career",
    "avg_miss_distance_in_zone_career",
    "avg_miss_distance_chase_career",
    # pitcher aggregate induced rates
    "pitcher_induced_swing_rate",
    "pitcher_induced_chase_rate",
    "pitcher_induced_contact_rate",
    "pitcher_induced_miss_rate",
    # pitcher x pitch type induced rates
    "pitcher_swing_rate_this_pitch",
    "pitcher_chase_rate_this_pitch",
    "pitcher_contact_rate_this_pitch",
    "pitcher_miss_rate_this_pitch",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "pitch_result_target"
TRAIN_FRACTION = 0.8


def time_based_split(df):
    df = df.sort_values("game_date").reset_index(drop=True)
    unique_dates = df["game_date"].drop_duplicates().sort_values().reset_index(drop=True)
    cutoff_date = unique_dates.iloc[int(len(unique_dates) * TRAIN_FRACTION)]
    train_df = df[df["game_date"] < cutoff_date]
    test_df = df[df["game_date"] >= cutoff_date]
    return train_df, test_df, cutoff_date


def prepare_features(df):
    X = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def print_confusion_matrix(y_test, preds, label_encoder):
    cm = confusion_matrix(y_test, preds, labels=list(range(len(label_encoder.classes_))))
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(cm_df)


def main():
    df = load_model1_features()
    train_df, test_df, cutoff_date = time_based_split(df)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows | cutoff: {cutoff_date.date()}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[TARGET_COLUMN])
    y_test = label_encoder.transform(test_df[TARGET_COLUMN])
    num_class = len(label_encoder.classes_)

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    mlflow.set_experiment("model1_plate_discipline")

    # XGBoost
    with mlflow.start_run(run_name="xgboost"):
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
        xgb_logloss = log_loss(y_test, xgb_proba)
        mlflow.log_metric("accuracy", xgb_accuracy)
        mlflow.log_metric("log_loss", xgb_logloss)
        mlflow.xgboost.log_model(xgb_model, "model", model_format="json")

        print("\n=== XGBoost ===")
        print(f"Accuracy: {xgb_accuracy:.4f} | Log Loss: {xgb_logloss:.4f}")
        print(classification_report(y_test, xgb_preds, target_names=label_encoder.classes_))
        print_confusion_matrix(y_test, xgb_preds, label_encoder)

    # LightGBM
    with mlflow.start_run(run_name="lightgbm"):
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
        lgb_model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

        lgb_preds = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)

        lgb_accuracy = accuracy_score(y_test, lgb_preds)
        lgb_logloss = log_loss(y_test, lgb_proba)
        mlflow.log_metric("accuracy", lgb_accuracy)
        mlflow.log_metric("log_loss", lgb_logloss)
        mlflow.lightgbm.log_model(lgb_model, "model")

        print("\n=== LightGBM ===")
        print(f"Accuracy: {lgb_accuracy:.4f} | Log Loss: {lgb_logloss:.4f}")
        print(classification_report(y_test, lgb_preds, target_names=label_encoder.classes_))
        print_confusion_matrix(y_test, lgb_preds, label_encoder)


if __name__ == "__main__":
    main()
