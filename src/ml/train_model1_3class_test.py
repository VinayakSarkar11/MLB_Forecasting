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
from sklearn.preprocessing import LabelEncoder

from src.ml.load_data import load_model1_features
from src.ml.train_model1 import prepare_features, time_based_split

# Collapses the 6-class target into 3 classes, splitting foul_tip out of
# Swinging_Strike and into Contact — tests whether the model's weakness on
# Swinging_Strike/Foul/In_Play is really just "did the batter make contact"
# vs. having to also separate contact outcomes from each other.
#
# hit_by_pitch is dropped — it doesn't fit any of the three buckets.

CONTACT_DESCRIPTIONS = {"hit_into_play", "foul", "foul_bunt", "foul_tip", "bunt_foul_tip"}
STRIKE_DESCRIPTIONS = {"called_strike", "swinging_strike", "swinging_strike_blocked", "missed_bunt"}
BALL_DESCRIPTIONS = {"ball", "blocked_ball", "pitchout"}


def build_3class_target(df):
    def label(description):
        if description in CONTACT_DESCRIPTIONS:
            return "Contact"
        if description in STRIKE_DESCRIPTIONS:
            return "Strike"
        if description in BALL_DESCRIPTIONS:
            return "Ball"
        return None

    df = df.copy()
    df["target_3class"] = df["description"].map(label)
    return df[df["target_3class"].notna()]


def main():
    df = load_model1_features()
    df = build_3class_target(df)
    train_df, test_df, cutoff_date = time_based_split(df)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows | cutoff: {cutoff_date.date()}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["target_3class"])
    y_test = label_encoder.transform(test_df["target_3class"])
    num_class = len(label_encoder.classes_)

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    mlflow.set_experiment("model1_plate_discipline_3class_test")

    with mlflow.start_run(run_name="xgboost_3class"):
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

        print("\n=== XGBoost (3-class) ===")
        print(f"Accuracy: {xgb_accuracy:.4f} | Log Loss: {xgb_logloss:.4f}")
        print(classification_report(y_test, xgb_preds, target_names=label_encoder.classes_))

    with mlflow.start_run(run_name="lightgbm_3class"):
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
        lgb_model.fit(X_train, y_train, categorical_feature=[c for c in X_train.columns if X_train[c].dtype.name == "category"])

        lgb_preds = lgb_model.predict(X_test)
        lgb_proba = lgb_model.predict_proba(X_test)

        lgb_accuracy = accuracy_score(y_test, lgb_preds)
        lgb_logloss = log_loss(y_test, lgb_proba)
        mlflow.log_metric("accuracy", lgb_accuracy)
        mlflow.log_metric("log_loss", lgb_logloss)
        mlflow.lightgbm.log_model(lgb_model, "model")

        print("\n=== LightGBM (3-class) ===")
        print(f"Accuracy: {lgb_accuracy:.4f} | Log Loss: {lgb_logloss:.4f}")
        print(classification_report(y_test, lgb_preds, target_names=label_encoder.classes_))


if __name__ == "__main__":
    main()
