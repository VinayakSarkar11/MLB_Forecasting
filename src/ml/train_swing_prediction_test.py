import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, log_loss

from src.ml.load_data import load_model1_features
from src.ml.train_model1 import prepare_features, time_based_split

# Binary target: did the batter swing at this pitch? Mirrors the is_swing
# flag in 08_model1_final_view.sql's `flags` CTE exactly, so this stays
# consistent with how is_swing is defined for the rolling contact/swing
# rate features. Note: bunt_foul_tip is NOT included here, matching that
# CTE's definition (likely an oversight there, but kept consistent rather
# than silently diverging).
SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
    "hit_into_play",
}


def build_swing_target(df):
    df = df.copy()
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    return df


def main():
    df = load_model1_features()
    df = build_swing_target(df)
    train_df, test_df, cutoff_date = time_based_split(df)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows | cutoff: {cutoff_date.date()}")
    print(f"Swing rate — train: {train_df['is_swing'].mean():.3f} | test: {test_df['is_swing'].mean():.3f}")

    y_train = train_df["is_swing"].to_numpy()
    y_test = test_df["is_swing"].to_numpy()

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    mlflow.set_experiment("swing_prediction_test")

    with mlflow.start_run(run_name="xgboost_swing"):
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
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

        print("\n=== XGBoost (swing prediction) ===")
        print(f"Accuracy: {xgb_accuracy:.4f} | Log Loss: {xgb_logloss:.4f}")
        print(classification_report(y_test, xgb_preds, target_names=["No_Swing", "Swing"]))

    with mlflow.start_run(run_name="lightgbm_swing"):
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
        }
        mlflow.log_params(lgb_params)

        categorical_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train, categorical_feature=categorical_cols)

        lgb_preds = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)

        lgb_accuracy = accuracy_score(y_test, lgb_preds)
        lgb_logloss = log_loss(y_test, lgb_proba)
        mlflow.log_metric("accuracy", lgb_accuracy)
        mlflow.log_metric("log_loss", lgb_logloss)
        mlflow.lightgbm.log_model(lgb_model, "model")

        print("\n=== LightGBM (swing prediction) ===")
        print(f"Accuracy: {lgb_accuracy:.4f} | Log Loss: {lgb_logloss:.4f}")
        print(classification_report(y_test, lgb_preds, target_names=["No_Swing", "Swing"]))


if __name__ == "__main__":
    main()
