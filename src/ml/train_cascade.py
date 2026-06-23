import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from src.ml.load_data import load_model1_features
from src.ml.train_model1 import prepare_features, time_based_split

MODEL_DIR = Path("models/cascade")

SWING_OUTCOMES = {"Swinging_Strike", "Foul", "In_Play"}
CONTACT_OUTCOMES = {"Foul", "In_Play"}

CLASSES = ["Ball", "Called_Strike", "Foul", "In_Play", "Swinging_Strike"]


def build_cascade_labels(df):
    df = df.copy()
    df["is_swing"] = df["pitch_result_target"].isin(SWING_OUTCOMES).astype(int)
    df["is_contact"] = df["pitch_result_target"].isin(CONTACT_OUTCOMES).astype(int)
    df["zone_in"] = df["zone"].between(1, 9).astype(int)
    return df


def train_stage(stage_name, X_train, y_train, model_type="lgb", class_weight=None):
    """Train one binary stage of the cascade.

    class_weight: None → no weighting
                  "balanced" → sklearn proportional balance
                  float > 1  → minority class gets that many times the majority weight
    """
    if class_weight == "balanced":
        sample_weight = compute_sample_weight("balanced", y_train)
    elif isinstance(class_weight, (int, float)):
        minority = np.bincount(y_train.astype(int)).argmin()
        sample_weight = np.where(y_train == minority, float(class_weight), 1.0)
    else:
        sample_weight = None

    if model_type == "lgb":
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "max_depth": 6,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "verbosity": -1,
        }
        categorical_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
        model = lgb.LGBMClassifier(**params)

        with mlflow.start_run(run_name=stage_name, nested=True):
            mlflow.log_params({"model_type": model_type, **params})
            model.fit(X_train, y_train, sample_weight=sample_weight, categorical_feature=categorical_cols)
            mlflow.lightgbm.log_model(model, "model")

    else:  # xgb
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "enable_categorical": True,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
        }
        model = xgb.XGBClassifier(**params)

        with mlflow.start_run(run_name=stage_name, nested=True):
            mlflow.log_params({"model_type": model_type, **params})
            model.fit(X_train, y_train, sample_weight=sample_weight)
            mlflow.xgboost.log_model(model, "model", model_format="json")

    return model


def predict_cascade(X, stage0, stage1, stage2a, stage2b, stage3, swing_threshold=0.55):
    """Apply all five stages in sequence and return a 5-class prediction Series."""
    zone_proba = stage0.predict_proba(X)[:, 1]
    X_aug = X.assign(pred_zone_in_prob=zone_proba)

    swing_proba = stage1.predict_proba(X_aug)[:, 1]
    swing_pred = (swing_proba >= swing_threshold).astype(int)

    final = pd.Series("", index=X.index, dtype=object)

    no_swing_idx = X.index[swing_pred == 0]
    swing_idx = X.index[swing_pred == 1]

    if len(no_swing_idx):
        zone_pred = stage2a.predict(X_aug.loc[no_swing_idx])
        final.loc[no_swing_idx] = np.where(zone_pred == 0, "Ball", "Called_Strike")

    if len(swing_idx):
        contact_pred = stage2b.predict(X_aug.loc[swing_idx])
        miss_idx = swing_idx[contact_pred == 0]
        contact_idx = swing_idx[contact_pred == 1]
        final.loc[miss_idx] = "Swinging_Strike"

        if len(contact_idx):
            foul_pred = stage3.predict(X_aug.loc[contact_idx])
            final.loc[contact_idx] = np.where(foul_pred == 0, "Foul", "In_Play")

    return final


def train_and_evaluate_cascade(model_type, X_train, train_df, X_test, test_df):
    """Train all four stages and evaluate the full cascade for one model type."""
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} cascade")
    print(f"{'='*50}")

    with mlflow.start_run(run_name=f"cascade_{model_type}", nested=True):

        print("Stage 0 — Zone In/Out predictor (pre-pitch location prior)...")
        stage0 = train_stage(f"stage0_location_{model_type}", X_train, train_df["zone_in"], model_type, class_weight="balanced")

        stage0_train_acc = (stage0.predict(X_train) == train_df["zone_in"].values).mean()
        stage0_test_acc  = (stage0.predict(X_test)  == test_df["zone_in"].values).mean()
        print(f"Stage 0 accuracy — train: {stage0_train_acc:.4f} | test: {stage0_test_acc:.4f}")

        # Augment feature matrices with Stage 0's zone probability.
        # Stage 0 is trained on X_train, so its in-sample predictions have slight optimism;
        # this is bounded because the model doesn't perfectly fit training data.
        X_train_aug = X_train.assign(pred_zone_in_prob=stage0.predict_proba(X_train)[:, 1])
        X_test_aug = X_test.assign(pred_zone_in_prob=stage0.predict_proba(X_test)[:, 1])

        print("Stage 1 — Swing vs No Swing...")
        stage1 = train_stage(f"stage1_swing_{model_type}", X_train_aug, train_df["is_swing"], model_type)

        # Stage 1 diagnostics
        swing_proba_test = stage1.predict_proba(X_test_aug)[:, 1]
        actual_swing = test_df["is_swing"].values
        for thresh in [0.40, 0.45, 0.50, 0.55]:
            pred_swing = (swing_proba_test >= thresh).astype(int)
            tp = ((pred_swing == 1) & (actual_swing == 1)).sum()
            fp = ((pred_swing == 1) & (actual_swing == 0)).sum()
            fn = ((pred_swing == 0) & (actual_swing == 1)).sum()
            tn = ((pred_swing == 0) & (actual_swing == 0)).sum()
            acc  = (tp + tn) / len(actual_swing)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            pct_predicted_swing = pred_swing.mean()
            print(f"  thresh={thresh:.2f}  acc={acc:.3f}  swing_precision={prec:.3f}  swing_recall={rec:.3f}  predicted_swing%={pct_predicted_swing:.3f}")

        importance = pd.Series(stage1.feature_importances_, index=X_train_aug.columns)
        print(f"\nStage 1 top-20 features ({model_type.upper()}):")
        print(importance.sort_values(ascending=False).head(20).to_string())
        null_rates = X_train_aug.isnull().mean().sort_values(ascending=False)
        print(f"\nTop-10 NULL rates in training features:")
        print(null_rates.head(10).to_string())
        print()

        print("Stage 2a — Ball vs Called Strike...")
        no_swing_mask = train_df["is_swing"] == 0
        y_zone = (train_df.loc[no_swing_mask, "pitch_result_target"] == "Called_Strike").astype(int)
        stage2a = train_stage(f"stage2a_zone_{model_type}", X_train_aug[no_swing_mask], y_zone, model_type, class_weight=1.2)

        print("Stage 2b — Contact vs Swinging Strike (class-balanced)...")
        swing_mask = train_df["is_swing"] == 1
        y_contact = train_df.loc[swing_mask, "is_contact"]
        stage2b = train_stage(f"stage2b_contact_{model_type}", X_train_aug[swing_mask], y_contact, model_type, class_weight=2.0)

        print("Stage 3 — Foul vs In Play...")
        contact_mask = (train_df["is_swing"] == 1) & (train_df["is_contact"] == 1)
        y_foul = (train_df.loc[contact_mask, "pitch_result_target"] == "In_Play").astype(int)
        stage3 = train_stage(f"stage3_foul_inplay_{model_type}", X_train_aug[contact_mask], y_foul, model_type)

        # Save
        save_dir = MODEL_DIR / model_type
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(stage0, save_dir / "stage0_location.pkl")
        joblib.dump(stage1, save_dir / "stage1_swing.pkl")
        joblib.dump(stage2a, save_dir / "stage2a_zone.pkl")
        joblib.dump(stage2b, save_dir / "stage2b_contact.pkl")
        joblib.dump(stage3, save_dir / "stage3_foul_inplay.pkl")
        print(f"Models saved to {save_dir}/")

        # Evaluate
        preds = predict_cascade(X_test, stage0, stage1, stage2a, stage2b, stage3, swing_threshold=0.50)
        true = test_df["pitch_result_target"]

        acc = accuracy_score(true, preds)
        mlflow.log_metric("cascade_accuracy", acc)

        print(f"\n=== {model_type.upper()} Cascade Evaluation ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(true, preds, labels=CLASSES, target_names=CLASSES))

        cm = confusion_matrix(true, preds, labels=CLASSES)
        cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
        print("Confusion matrix (rows = actual, columns = predicted):")
        print(cm_df)


def main():
    df = load_model1_features()
    df = df[df["pitch_result_target"] != "Hit_By_Pitch"].copy()
    df = build_cascade_labels(df)

    train_df, test_df, cutoff_date = time_based_split(df)
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | cutoff: {cutoff_date.date()}")

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)
    print(f"Feature matrix: {X_train.shape[1]} columns")  # expect 68 (pred_zone_in_prob added inside cascade)

    mlflow.set_experiment("cascade_model1")

    with mlflow.start_run(run_name="cascade_comparison"):
        train_and_evaluate_cascade("lgb", X_train, train_df, X_test, test_df)
        train_and_evaluate_cascade("xgb", X_train, train_df, X_test, test_df)

if __name__ == "__main__":
    main()
