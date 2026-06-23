"""
Isolated evaluation of Stage 0 (zone-in prior) + Stage 1 (swing/no-swing).

Usage:
    python -m src.ml.test_stage1
"""

import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from src.ml.load_data import load_model1_features
from src.ml.train_cascade import build_cascade_labels
from src.ml.train_model1 import prepare_features, time_based_split

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    max_depth=6,
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=500,
    verbosity=-1,
)


def train_lgb(X, y, sample_weight=None):
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X, y, sample_weight=sample_weight, categorical_feature=cat_cols)
    return m


def main():
    print("Loading data...")
    df = load_model1_features()
    df = df[df["pitch_result_target"] != "Hit_By_Pitch"].copy()
    df = build_cascade_labels(df)

    train_df, test_df, cutoff_date = time_based_split(df)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,} | cutoff: {cutoff_date.date()}")
    print(f"Actual swing rate — train: {train_df['is_swing'].mean():.3f} | test: {test_df['is_swing'].mean():.3f}\n")

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)
    print(f"Features: {X_train.shape[1]}\n")

    # Stage 0 — zone in/out
    print("Training Stage 0 (zone in/out)...")
    sw0 = compute_sample_weight("balanced", train_df["zone_in"])
    stage0 = train_lgb(X_train, train_df["zone_in"], sample_weight=sw0)
    s0_train = (stage0.predict(X_train) == train_df["zone_in"].values).mean()
    s0_test  = (stage0.predict(X_test)  == test_df["zone_in"].values).mean()
    print(f"Stage 0 accuracy — train: {s0_train:.4f} | test: {s0_test:.4f}\n")

    X_train_aug = X_train.assign(pred_zone_in_prob=stage0.predict_proba(X_train)[:, 1])
    X_test_aug  = X_test.assign(pred_zone_in_prob=stage0.predict_proba(X_test)[:, 1])

    # Stage 1 — swing vs no-swing (no class weighting — 52/48 near-balanced)
    print("Training Stage 1 (swing / no-swing)...")
    stage1 = train_lgb(X_train_aug, train_df["is_swing"])

    swing_proba = stage1.predict_proba(X_test_aug)[:, 1]
    actual      = test_df["is_swing"].values

    # Threshold sweep
    print("\nThreshold sweep:")
    header = f"  {'thresh':>6}  {'acc':>6}  {'precision':>9}  {'recall':>7}  {'pred_swing%':>11}  {'actual_swing%':>13}"
    print(header)
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        pred = (swing_proba >= thresh).astype(int)
        tp = int(((pred == 1) & (actual == 1)).sum())
        fp = int(((pred == 1) & (actual == 0)).sum())
        fn = int(((pred == 0) & (actual == 1)).sum())
        tn = int(((pred == 0) & (actual == 0)).sum())
        acc  = (tp + tn) / len(actual)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  {thresh:>6.2f}  {acc:>6.3f}  {prec:>9.3f}  {rec:>7.3f}  {pred.mean():>11.3f}  {actual.mean():>13.3f}")

    # Feature importance
    importance = pd.Series(stage1.feature_importances_, index=X_train_aug.columns)
    print(f"\nTop-20 features (Stage 1):")
    print(importance.sort_values(ascending=False).head(20).to_string())

    # Per-count breakdown at threshold=0.50
    print("\n\nSwing rate by count (actual vs predicted @ 0.50 threshold):")
    pred_50 = (swing_proba >= 0.50).astype(int)
    breakdown = (
        test_df[["balls", "strikes", "is_swing"]]
        .assign(pred_swing=pred_50, proba=swing_proba)
        .groupby(["balls", "strikes"])
        .agg(n=("is_swing", "count"),
             actual_rate=("is_swing", "mean"),
             pred_rate=("pred_swing", "mean"),
             avg_proba=("proba", "mean"))
        .round(3)
    )
    print(breakdown.to_string())

    # Calibration by decile
    print("\n\nCalibration (predicted probability deciles vs actual swing rate):")
    cal_df = pd.DataFrame({"pred": swing_proba, "actual": actual})
    cal_df["bin"] = pd.qcut(swing_proba, q=10, duplicates="drop")
    cal = cal_df.groupby("bin", observed=True).agg(
        n=("actual", "count"),
        mean_pred=("pred", "mean"),
        actual_rate=("actual", "mean"),
    ).round(3)
    print(cal.to_string())

    # NULL rates
    print("\n\nTop-10 NULL rates (train features):")
    null_rates = X_train_aug.isnull().mean().sort_values(ascending=False)
    print(null_rates.head(10).to_string())


if __name__ == "__main__":
    main()
