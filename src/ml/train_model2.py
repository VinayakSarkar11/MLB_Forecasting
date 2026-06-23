"""
Mid-AB plate-appearance outcome cascade (Model 2).

Each row is a PITCH in a PA, not just the final pitch. The final PA outcome is
the target for every row, but the within-AB features (balls, strikes, pitch
sequence) change pitch-by-pitch, letting the model update its prediction as the
at-bat progresses.

Stage 1  (binary):     In play vs Not in play           → P(in_play)
Stage 2a (regression): xwOBA on ball in play            → pred_xwOBA
Stage 2a (multiclass): single / xbh / home_run / out    → P(outcome | BIP)
Stage 2b (binary):     strikeout vs walk                → P(K | NIP)

Cascade output — full probability distribution over 6 outcomes + E[wOBA]:
    P(strikeout)   = P(NIP) × P(K | NIP)
    P(walk)        = P(NIP) × P(walk | NIP)
    P(single)      = P(IP)  × P(single | BIP)
    P(xbh)         = P(IP)  × P(xbh | BIP)
    P(home_run)    = P(IP)  × P(HR | BIP)
    P(out_in_play) = P(IP)  × P(out | BIP)
    E[wOBA]        = P(IP) × pred_xwOBA + P(NIP) × P(walk|NIP) × WALK_WOBA

Usage:
    python -m src.ml.train_model2
"""

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
import xgboost as xgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, mean_squared_error,
    roc_auc_score, brier_score_loss, log_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.ml.load_data import load_model2_midab_features
from src.ml.train_model1 import time_based_split

MODEL_DIR = Path("models/cascade_midab")

IN_PLAY_OUTCOMES     = {"single", "xbh", "home_run", "out_in_play"}
NOT_IN_PLAY_OUTCOMES = {"strikeout", "walk"}
IN_PLAY_CLASSES      = ["single", "xbh", "home_run", "out_in_play"]
ALL_OUTCOMES         = ["strikeout", "walk", "single", "xbh", "home_run", "out_in_play"]

WALK_WOBA = 0.690   # wOBA linear weight for walk/HBP (Statcast era)

# Stage 2a multiclass weights — chosen by prior val-fold sweep:
# [1.0, 3.0, 8.5, 13.0] had best macro F1 without collapsing to all-HR predictions.
S2A_CLS_WEIGHTS = {"out_in_play": 1.0, "single": 3.0, "xbh": 8.5, "home_run": 16.0}

NUMERIC_FEATURES = [
    # park factors (rolling rates at this home_team's park prior to this game)
    "park_hr_rate",
    "park_k_rate",
    "park_bb_rate",
    # situational
    "inning",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "num_runners_on",
    "runner_in_scoring_position",
    "bat_score_diff",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
    # batter career outcomes
    "batter_career_woba",
    "batter_career_k_rate",
    "batter_career_bb_rate",
    "batter_career_hbp_rate",
    "batter_career_hr_rate",
    "batter_career_babip",
    "batter_career_avg_exit_velo",
    "batter_career_avg_launch_angle",
    "batter_career_hard_hit_rate",
    "batter_career_gb_rate",
    "batter_career_ld_rate",
    "batter_career_babip_luck",   # career_babip - xBABIP from bb-type mix; skill vs variance
    "batter_career_avg_xwoba",
    # batter 30d outcomes
    "batter_30d_woba",
    "batter_30d_k_rate",
    "batter_30d_bb_rate",
    "batter_30d_hr_rate",
    "batter_30d_babip",
    "batter_30d_avg_exit_velo",
    "batter_30d_avg_launch_angle",
    "batter_30d_avg_xwoba",
    # batter swing behavior
    "batter_career_contact_rate",
    "batter_career_swing_rate",
    # K:BB ratios
    "batter_career_k_bb_ratio",
    "pitcher_career_k_bb_ratio",
    "batter_career_k_bb_ratio_vs_rhp",
    "batter_career_k_bb_ratio_vs_lhp",
    "pitcher_career_k_bb_ratio_vs_rhb",
    "pitcher_career_k_bb_ratio_vs_lhb",
    # batter platoon splits (career)
    "batter_career_k_rate_vs_rhp",
    "batter_career_k_rate_vs_lhp",
    "batter_career_bb_rate_vs_rhp",
    "batter_career_bb_rate_vs_lhp",
    "batter_career_woba_vs_rhp",
    "batter_career_woba_vs_lhp",
    "batter_career_avg_exit_velo_vs_rhp",
    "batter_career_avg_exit_velo_vs_lhp",
    "batter_career_avg_xwoba_vs_rhp",
    "batter_career_avg_xwoba_vs_lhp",
    # pitcher career outcomes
    "pitcher_career_woba_against",
    "pitcher_career_k_rate",
    "pitcher_career_bb_rate",
    "pitcher_career_hr_rate",
    "pitcher_career_babip_against",
    "pitcher_career_gb_rate",
    # pitcher platoon splits (career)
    "pitcher_career_k_rate_vs_rhb",
    "pitcher_career_k_rate_vs_lhb",
    "pitcher_career_bb_rate_vs_rhb",
    "pitcher_career_bb_rate_vs_lhb",
    "pitcher_career_woba_against_vs_rhb",
    "pitcher_career_woba_against_vs_lhb",
    # pitcher contact quality by pitch category
    "pitcher_career_avg_exit_velo_fastball",
    "pitcher_career_avg_exit_velo_breaking",
    "pitcher_career_avg_exit_velo_offspeed",
    "pitcher_career_avg_xwoba_fastball",
    "pitcher_career_avg_xwoba_breaking",
    "pitcher_career_avg_xwoba_offspeed",
    # pitcher 30d outcomes
    "pitcher_30d_woba_against",
    "pitcher_30d_k_rate",
    "pitcher_30d_bb_rate",
    # pitcher pitch rates
    "pitcher_career_swing_rate",
    "pitcher_career_miss_rate",
    "pitcher_career_avg_velo",
    # platoon interaction features
    # same_hand = 1 when pitcher and batter share handedness (pitcher advantage)
    # split differentials encode the magnitude of each side's platoon tendency
    "same_hand",
    "batter_platoon_woba_split",
    "pitcher_platoon_woba_split",
    # within-AB state (count + pitch sequence before current pitch)
    "balls",
    "strikes",
    "pitches_seen_in_ab",
    "fastballs_seen_in_ab",
    "breaking_seen_in_ab",
    "offspeed_seen_in_ab",
    "fouls_in_ab",
    "whiffs_in_ab",
]

CATEGORICAL_FEATURES = [
    "stand",
    "p_throws",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "home_team",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def build_pa_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_in_play"] = df["pa_outcome"].isin(IN_PLAY_OUTCOMES).astype(int)
    return df


# ── Model parameter helpers ───────────────────────────────────────────────────

def _lgb_binary():
    return dict(objective="binary", metric="binary_logloss",
                max_depth=6, num_leaves=63, learning_rate=0.05,
                n_estimators=500, verbosity=-1)

def _xgb_binary():
    return dict(objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", enable_categorical=True,
                max_depth=6, learning_rate=0.05, n_estimators=500)

def _lgb_multiclass(n):
    return dict(objective="multiclass", num_class=n, metric="multi_logloss",
                max_depth=6, num_leaves=63, learning_rate=0.05,
                n_estimators=500, verbosity=-1)

def _xgb_multiclass(n):
    return dict(objective="multi:softprob", num_class=n, eval_metric="mlogloss",
                tree_method="hist", enable_categorical=True,
                max_depth=6, learning_rate=0.05, n_estimators=500)

def _lgb_regression():
    return dict(objective="regression", metric="rmse",
                max_depth=6, num_leaves=63, learning_rate=0.05,
                n_estimators=500, verbosity=-1)

def _xgb_regression():
    return dict(objective="reg:squarederror", eval_metric="rmse",
                tree_method="hist", enable_categorical=True,
                max_depth=6, learning_rate=0.05, n_estimators=500)

def _cat_cols(X):
    return [c for c in X.columns if X[c].dtype.name == "category"]


# ── Shared helpers ───────────────────────────────────────────────────────────

def _sit_key(df):
    """Situation key: count + outs + exact base state (288 combinations)."""
    return (
        df["balls"].astype(str) + "-" + df["strikes"].astype(str)
        + "_o" + df["outs_when_up"].astype(str)
        + "_b" + df["on_1b"].astype(int).astype(str)
               + df["on_2b"].astype(int).astype(str)
               + df["on_3b"].astype(int).astype(str)
    ).values


def _sit_naive_proba(train_df, test_df, labels, y_train=None):
    """Build situation-naive probability matrix from training data.
    y_train defaults to train_df['pa_outcome'] if not provided."""
    y_tr = y_train if y_train is not None else train_df["pa_outcome"].values
    tr_key  = _sit_key(train_df)
    te_key  = _sit_key(test_df)
    sit_base = (
        pd.DataFrame({"_sk": tr_key, "_y": y_tr})
        .groupby("_sk")["_y"]
        .value_counts(normalize=True)
        .unstack(fill_value=1e-9)
        .reindex(columns=labels, fill_value=1e-9)
    )
    overall = np.array([(y_tr == c).mean() for c in labels])
    unseen  = [k for k in te_key if k not in sit_base.index]
    if unseen:
        sit_base = pd.concat([sit_base, pd.DataFrame(
            [overall] * len(set(unseen)), index=list(set(unseen)), columns=labels
        )])
    proba = sit_base.loc[te_key].values.astype(float)
    return proba / proba.sum(axis=1, keepdims=True)


# ── Stage training helpers ────────────────────────────────────────────────────

def train_binary_stage(name, X, y, model_type, class_weight="balanced"):
    sw = compute_sample_weight("balanced", y) if class_weight == "balanced" else None
    if model_type == "lgb":
        m = lgb.LGBMClassifier(**_lgb_binary())
        with mlflow.start_run(run_name=name, nested=True):
            m.fit(X, y, sample_weight=sw, categorical_feature=_cat_cols(X))
            mlflow.lightgbm.log_model(m, "model")
    else:
        m = xgb.XGBClassifier(**_xgb_binary())
        with mlflow.start_run(run_name=name, nested=True):
            m.fit(X, y, sample_weight=sw)
            mlflow.xgboost.log_model(m, "model", model_format="json")
    return m


def train_multiclass_stage(name, X, y_raw, classes, model_type, sample_weight=None):
    le = LabelEncoder()
    le.fit(classes)
    y = le.transform(y_raw)
    n = len(classes)
    if model_type == "lgb":
        m = lgb.LGBMClassifier(**_lgb_multiclass(n))
        with mlflow.start_run(run_name=name, nested=True):
            m.fit(X, y, sample_weight=sample_weight, categorical_feature=_cat_cols(X))
            mlflow.lightgbm.log_model(m, "model")
    else:
        m = xgb.XGBClassifier(**_xgb_multiclass(n))
        with mlflow.start_run(run_name=name, nested=True):
            m.fit(X, y, sample_weight=sample_weight)
            mlflow.xgboost.log_model(m, "model", model_format="json")
    return m, le


def train_regression_stage(name, X, y, model_type):
    if model_type == "lgb":
        m = lgb.LGBMRegressor(**_lgb_regression())
        with mlflow.start_run(run_name=name, nested=True):
            m.fit(X, y, categorical_feature=_cat_cols(X))
            mlflow.lightgbm.log_model(m, "model")
    else:
        m = xgb.XGBRegressor(**_xgb_regression())
        with mlflow.start_run(run_name=name, nested=True):
            m.fit(X, y)
            mlflow.xgboost.log_model(m, "model", model_format="json")
    return m


# ── Cascade prediction ────────────────────────────────────────────────────────

def predict_cascade(X, stage1, stage2a_reg, stage2a_cls, stage2a_le, stage2a_calibrators,
                    stage2b, stage2b_calibrator):
    """Return a DataFrame with one column per outcome + E[wOBA].

    Columns: strikeout, walk, single, xbh, home_run, out_in_play, expected_woba
    """
    p_ip   = stage1.predict_proba(X)[:, 1]
    p_nip  = 1.0 - p_ip

    raw_k  = stage2b.predict_proba(X)[:, 1]
    p_k    = stage2b_calibrator.transform(raw_k)
    p_walk = 1.0 - p_k

    # Stage 2a classifier: raw proba → isotonic calibration → empirical rates
    raw_bip  = stage2a_cls.predict_proba(X)
    cal_bip  = apply_calibrators(raw_bip, stage2a_calibrators, stage2a_le)
    bip_p = {}
    for i, cls_name in enumerate(stage2a_le.classes_):
        bip_p[cls_name] = cal_bip[:, i]

    pred_xwoba = stage2a_reg.predict(X)

    out = pd.DataFrame({
        "strikeout":   p_nip * p_k,
        "walk":        p_nip * p_walk,
        "single":      p_ip  * bip_p["single"],
        "xbh":         p_ip  * bip_p["xbh"],
        "home_run":    p_ip  * bip_p["home_run"],
        "out_in_play": p_ip  * bip_p["out_in_play"],
        "expected_woba": p_ip * pred_xwoba + p_nip * p_walk * WALK_WOBA,
    }, index=X.index)

    return out


# ── Probability calibration ───────────────────────────────────────────────────

def fit_calibrators(raw_proba, y_true_labels, le):
    """Fit per-class IsotonicRegression to remap inflated minority-class probabilities.

    Training with imbalanced sample weights shifts raw P(class) upward for rare classes.
    Isotonic regression maps raw → empirical rate while preserving ordering.

    Returns dict: {cls_name: fitted IsotonicRegression}
    """
    calibrators = {}
    for i, cls_name in enumerate(le.classes_):
        y_bin = (y_true_labels == cls_name).astype(int)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(raw_proba[:, i], y_bin)
        calibrators[cls_name] = ir
    return calibrators


def apply_calibrators(raw_proba, calibrators, le):
    """Apply per-class calibrators then renormalise rows to sum to 1."""
    cal = np.empty_like(raw_proba)
    for i, cls_name in enumerate(le.classes_):
        cal[:, i] = calibrators[cls_name].transform(raw_proba[:, i])
    row_sums = np.maximum(cal.sum(axis=1, keepdims=True), 1e-10)
    return cal / row_sums


# ── Main training loop ────────────────────────────────────────────────────────

def train_and_evaluate(model_type, X_train, train_df, X_test, test_df):
    print(f"\n{'='*50}\nTraining {model_type.upper()} cascade\n{'='*50}")

    with mlflow.start_run(run_name=f"midab_cascade_{model_type}", nested=True):

        # Hold out 15% of training rows for cascade E[wOBA] calibration.
        # Stages are trained on the 85% fit set so the calibration holdout is
        # truly unseen. The Stage 2a classifier further splits the fit set's
        # in-play rows 85/15 for its own isotonic calibration.
        woba_cal_cut  = int(len(X_train) * 0.85)
        X_tr_fit      = X_train.iloc[:woba_cal_cut]
        train_df_fit  = train_df.iloc[:woba_cal_cut]
        X_woba_cal    = X_train.iloc[woba_cal_cut:]
        train_df_wcal = train_df.iloc[woba_cal_cut:]

        # ── Stage 1: in play vs not in play ──────────────────────────────────
        print("Stage 1: in play vs not in play...")
        s1 = train_binary_stage(f"s1_inplay_{model_type}", X_tr_fit, train_df_fit["is_in_play"],
                                model_type, class_weight=None)
        s1_acc = accuracy_score(test_df["is_in_play"], s1.predict(X_test))
        naive  = max(test_df["is_in_play"].mean(), 1 - test_df["is_in_play"].mean())
        print(f"  test accuracy: {s1_acc:.4f}  (naive: {naive:.4f})")

        # ── Stage 2a — regression (xwOBA on BIP) ─────────────────────────────
        print("Stage 2a regression: xwOBA on balls in play...")
        ip_mask_tr = train_df_fit["is_in_play"] == 1
        y_xwoba_tr = train_df_fit.loc[ip_mask_tr, "bip_xwoba"]
        valid_tr   = y_xwoba_tr.notna()
        s2a_reg = train_regression_stage(
            f"s2a_xwoba_{model_type}",
            X_tr_fit[ip_mask_tr][valid_tr],
            y_xwoba_tr[valid_tr],
            model_type,
        )
        ip_mask_te = test_df["is_in_play"] == 1
        y_xwoba_te = test_df.loc[ip_mask_te, "bip_xwoba"]
        valid_te   = y_xwoba_te.notna()
        pred_xwoba = s2a_reg.predict(X_test[ip_mask_te][valid_te])
        act_xwoba  = y_xwoba_te[valid_te].values
        rmse       = np.sqrt(mean_squared_error(act_xwoba, pred_xwoba))
        naive_rmse = np.sqrt(mean_squared_error(act_xwoba, np.full_like(pred_xwoba, act_xwoba.mean())))
        print(f"  RMSE: {rmse:.4f}  (naive RMSE: {naive_rmse:.4f})")

        # ── Stage 2a — multiclass (P(outcome | BIP)) ─────────────────────────
        # Train on ALL in-play fit rows. Aggressive weights teach the model which
        # features predict rare classes. Per-class isotonic calibrators are fit on
        # the same 15% cascade holdout (X_woba_cal) — no additional data sacrifice.
        # The two calibrations are independent: per-class calibrators act on bip_proba
        # from the multiclass model; the cascade wOBA calibrator acts on pred_xwoba
        # from the regression. Sharing the holdout maximises classifier training data.
        print("Stage 2a classifier: P(single/xbh/HR/out | ball in play)...")
        X_tr_ip = X_tr_fit[ip_mask_tr]
        y_tr_ip = train_df_fit.loc[ip_mask_tr, "pa_outcome"]

        sw_cls = y_tr_ip.map(S2A_CLS_WEIGHTS).values
        s2a_cls, s2a_le = train_multiclass_stage(
            f"s2a_cls_{model_type}",
            X_tr_ip, y_tr_ip,
            IN_PLAY_CLASSES, model_type,
            sample_weight=sw_cls,
        )

        # Fit per-class isotonic calibrators on the cascade wOBA holdout (in-play rows)
        ip_mask_wcal    = train_df_wcal["is_in_play"] == 1
        raw_cal_proba   = s2a_cls.predict_proba(X_woba_cal[ip_mask_wcal])
        s2a_calibrators = fit_calibrators(raw_cal_proba,
                                          train_df_wcal.loc[ip_mask_wcal, "pa_outcome"].values,
                                          s2a_le)

        # Apply calibration to test set
        raw_proba_te = s2a_cls.predict_proba(X_test[ip_mask_te])
        bip_proba_te = apply_calibrators(raw_proba_te, s2a_calibrators, s2a_le)

        cls_names  = list(s2a_le.classes_)
        hr_col     = cls_names.index("home_run")
        single_col = cls_names.index("single")
        xbh_col    = cls_names.index("xbh")

        cal_df = test_df[ip_mask_te].copy()
        cal_df["pred_p_hr"]     = bip_proba_te[:, hr_col]
        cal_df["pred_p_single"] = bip_proba_te[:, single_col]
        cal_df["pred_p_xbh"]    = bip_proba_te[:, xbh_col]
        cal_df["actual_hr"]     = (cal_df["pa_outcome"] == "home_run").astype(int)
        cal_df["actual_single"] = (cal_df["pa_outcome"] == "single").astype(int)
        cal_df["actual_xbh"]    = (cal_df["pa_outcome"] == "xbh").astype(int)

        cal = (
            cal_df
            .assign(hr_q=pd.qcut(
                cal_df["batter_career_hr_rate"].fillna(cal_df["batter_career_hr_rate"].median()),
                q=5, duplicates="drop"
            ))
            .groupby("hr_q", observed=True)
            .agg(n=("actual_hr", "count"),
                 actual_hr_rate=("actual_hr", "mean"),
                 pred_p_hr=("pred_p_hr", "mean"),
                 actual_single_rate=("actual_single", "mean"),
                 pred_p_single=("pred_p_single", "mean"))
            .round(3)
        )
        print("  Probability calibration by batter career HR-rate quintile:")
        print(cal.to_string())

        # ── Stage 2b: strikeout vs walk ───────────────────────────────────────
        # Balanced class weights give the model discrimination power on the
        # minority walk class (29% base rate), but push raw P(K|NIP) toward 0.5
        # instead of the empirical 71%. Isotonic calibration on the cascade holdout
        # remaps raw P(K|NIP) back to empirical rates, fixing the walk over-prediction.
        print("Stage 2b: strikeout vs walk...")
        nip_mask_tr  = train_df_fit["is_in_play"] == 0
        y_k          = (train_df_fit.loc[nip_mask_tr, "pa_outcome"] == "strikeout").astype(int)
        s2b = train_binary_stage(f"s2b_notinplay_{model_type}", X_tr_fit[nip_mask_tr], y_k,
                                 model_type, class_weight="balanced")

        nip_mask_wcal   = train_df_wcal["is_in_play"] == 0
        raw_k_wcal      = s2b.predict_proba(X_woba_cal[nip_mask_wcal])[:, 1]
        y_k_wcal        = (train_df_wcal.loc[nip_mask_wcal, "pa_outcome"] == "strikeout").astype(int).values
        s2b_calibrator  = IsotonicRegression(out_of_bounds="clip")
        s2b_calibrator.fit(raw_k_wcal, y_k_wcal)

        nip_mask_te  = test_df["is_in_play"] == 0
        y_k_test     = (test_df.loc[nip_mask_te, "pa_outcome"] == "strikeout").astype(int)
        k_proba_test = s2b_calibrator.transform(s2b.predict_proba(X_test[nip_mask_te])[:, 1])
        raw_k_mean = s2b.predict_proba(X_test[nip_mask_te])[:, 1].mean()
        cal_k_mean = k_proba_test.mean()
        print(f"  Stage 2b calibration — raw P(K|NIP): {raw_k_mean:.3f}  "
              f"calibrated: {cal_k_mean:.3f}  actual: {y_k_test.mean():.3f}")
        print("  threshold sweep (on calibrated probabilities):")
        for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            pred_k = (k_proba_test >= t).astype(int)
            k_rec  = pred_k[y_k_test.values == 1].mean()
            w_rec  = (1 - pred_k[y_k_test.values == 0]).mean()
            print(f"    t={t:.2f}  K_recall={k_rec:.3f}  walk_recall={w_rec:.3f}")

        # ── Cascade E[wOBA] calibration ───────────────────────────────────────
        # The cascade systematically over-predicts E[wOBA] because aggressive
        # Stage 2a weights inflate P(HR/single/xbh) even after per-class isotonic
        # calibration. Fit a monotonic remapping from predicted → actual wOBA on
        # the held-out 15% of training rows (unseen by all stages).
        print("Cascade E[wOBA] calibration...")
        cascade_wcal   = predict_cascade(X_woba_cal, s1, s2a_reg, s2a_cls, s2a_le,
                                         s2a_calibrators, s2b, s2b_calibrator)
        actual_woba_wcal = train_df_wcal["actual_woba_value"].astype(float).values
        woba_calibrator  = IsotonicRegression(out_of_bounds="clip")
        woba_calibrator.fit(cascade_wcal["expected_woba"].values, actual_woba_wcal)
        raw_mean  = cascade_wcal["expected_woba"].mean()
        cal_mean  = woba_calibrator.transform(cascade_wcal["expected_woba"].values).mean()
        act_mean  = actual_woba_wcal.mean()
        print(f"  calibration holdout — raw: {raw_mean:.4f}  calibrated: {cal_mean:.4f}  "
              f"actual: {act_mean:.4f}")

        # ── Save models ───────────────────────────────────────────────────────
        save_dir = MODEL_DIR / model_type
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(s1,               save_dir / "stage1_inplay.pkl")
        joblib.dump(s2a_reg,          save_dir / "stage2a_xwoba_reg.pkl")
        joblib.dump(s2a_cls,          save_dir / "stage2a_cls.pkl")
        joblib.dump(s2a_le,           save_dir / "stage2a_le.pkl")
        joblib.dump(s2a_calibrators,  save_dir / "stage2a_calibrators.pkl")
        joblib.dump(s2b,              save_dir / "stage2b_strikeout_walk.pkl")
        joblib.dump(s2b_calibrator,   save_dir / "stage2b_calibrator.pkl")
        joblib.dump(woba_calibrator,  save_dir / "cascade_woba_calibrator.pkl")
        print(f"\nModels saved to {save_dir}/")

        # ── Full cascade: outcome probability distribution ─────────────────────
        cascade = predict_cascade(X_test, s1, s2a_reg, s2a_cls, s2a_le, s2a_calibrators,
                                  s2b, s2b_calibrator)
        cascade["expected_woba"] = woba_calibrator.transform(cascade["expected_woba"].values)
        actual_woba = test_df["actual_woba_value"].astype(float).values

        corr = np.corrcoef(cascade["expected_woba"].values, actual_woba)[0, 1]
        mlflow.log_metric("cascade_woba_corr", corr)

        print(f"\n=== {model_type.upper()} Cascade ===")
        print(f"  E[wOBA] correlation vs actual: {corr:.4f}")
        print(f"  Mean predicted: {cascade['expected_woba'].mean():.4f} | "
              f"Mean actual: {actual_woba.mean():.4f}")

        # ── Probabilistic scoring metrics ─────────────────────────────────────
        # AUC:  discrimination — does the model rank the right PAs highest?
        # BSS:  Brier Skill Score = 1 - BS/BS_naive — lift over always-predicting
        #       the empirical base rate. Positive = better than naive.
        # Log loss: full 6-class distribution quality; lower is better.
        #       Baseline = always predicting empirical class frequencies.
        print(f"\n  Probabilistic scoring metrics:")
        print(f"  {'Outcome':<12} {'Base rate':>10} {'AUC':>8} {'BSS':>8}")
        print(f"  {'-'*42}")

        # Build 6-class probability matrix in a fixed column order
        outcome_cols = ALL_OUTCOMES  # ["strikeout","walk","single","xbh","home_run","out_in_play"]
        for outcome in outcome_cols:
            y_bin      = (test_df["pa_outcome"].values == outcome).astype(int)
            pred_p     = cascade[outcome].values
            base_rate  = y_bin.mean()
            try:
                auc = roc_auc_score(y_bin, pred_p)
            except Exception:
                auc = float("nan")
            bs_model  = brier_score_loss(y_bin, pred_p)
            bs_naive  = brier_score_loss(y_bin, np.full(len(y_bin), base_rate))
            bss       = 1.0 - bs_model / bs_naive if bs_naive > 0 else float("nan")
            mlflow.log_metric(f"auc_{outcome}", auc)
            mlflow.log_metric(f"bss_{outcome}", bss)
            print(f"  {outcome:<12} {base_rate:>10.3f} {auc:>8.4f} {bss:>8.4f}")

        # Full 6-class log loss vs naive baseline (empirical base rates)
        prob_matrix   = cascade[outcome_cols].values
        # re-normalise in case calibration left tiny floating-point gaps
        prob_matrix   = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
        true_labels   = test_df["pa_outcome"].values
        ll_model      = log_loss(true_labels, prob_matrix, labels=outcome_cols)

        # Overall-rate naive (predicts same base rates for every row)
        base_rates = np.array([(true_labels == o).mean() for o in outcome_cols])
        ll_naive   = log_loss(true_labels,
                              np.tile(base_rates, (len(true_labels), 1)),
                              labels=outcome_cols)

        # Situation-aware naive: fit on training data, evaluated on test.
        # Measures whether knowing the specific pitcher/batter matchup adds value
        # beyond "runner on 2nd, 2 outs, 3-2 count."
        sit_key      = _sit_key(test_df)
        naive_sit_proba = _sit_naive_proba(train_df, test_df, outcome_cols)
        ll_naive_sit = log_loss(true_labels, naive_sit_proba, labels=outcome_cols)

        mlflow.log_metric("log_loss",            ll_model)
        mlflow.log_metric("log_loss_naive",       ll_naive)
        mlflow.log_metric("log_loss_sit_naive",   ll_naive_sit)
        print(f"\n  6-class log loss: {ll_model:.4f}")
        print(f"    vs overall-rate naive:   {ll_naive:.4f}  lift={ll_naive - ll_model:+.4f}")
        print(f"    vs situation naive:      {ll_naive_sit:.4f}  lift={ll_naive_sit - ll_model:+.4f}"
              "  ← correct comparison")

        # ── Simplified outcome splits ──────────────────────────────────────────
        # Aggregate probabilities into broader buckets. These show higher AUC/BSS
        # because merging rare classes removes aleatory noise — distinguishing
        # single vs XBH is much harder than distinguishing hit vs out.
        pa_out    = test_df["pa_outcome"].isin({"strikeout", "out_in_play"})
        pa_hit    = test_df["pa_outcome"].isin({"single", "xbh", "home_run"})
        pa_onbase = test_df["pa_outcome"].isin({"single", "xbh", "home_run", "walk"})

        p_out    = (cascade["strikeout"]  + cascade["out_in_play"]).values
        p_hit    = (cascade["single"] + cascade["xbh"] + cascade["home_run"]).values
        p_walk   = cascade["walk"].values
        p_onbase = p_hit + p_walk

        def _binary_metrics(y_bin, pred_p, label):
            br   = y_bin.mean()
            auc  = roc_auc_score(y_bin, pred_p)
            bs   = brier_score_loss(y_bin, pred_p)
            bsn  = brier_score_loss(y_bin, np.full(len(y_bin), br))
            bss  = 1.0 - bs / bsn
            return auc, bss, br

        print(f"\n  Simplified splits:")
        print(f"  {'Split':<18} {'Base rate':>10} {'AUC':>8} {'BSS':>8}")
        print(f"  {'-'*48}")

        for label, y_bin, pred_p in [
            ("on-base vs out",  pa_onbase.astype(int).values, p_onbase),
            ("hit vs not-hit",  pa_hit.astype(int).values,    p_hit),
            ("out vs not-out",  pa_out.astype(int).values,    p_out),
        ]:
            auc, bss, br = _binary_metrics(y_bin, pred_p, label)
            mlflow.log_metric(f"auc_{label.replace(' ', '_')}", auc)
            mlflow.log_metric(f"bss_{label.replace(' ', '_')}", bss)
            print(f"  {label:<18} {br:>10.3f} {auc:>8.4f} {bss:>8.4f}")

        # 3-class: out / walk / hit
        p3 = np.stack([p_out, p_walk, p_hit], axis=1)
        p3 = p3 / p3.sum(axis=1, keepdims=True)
        y3 = np.where(pa_hit.values, "hit", np.where(pa_out.values, "out", "walk"))
        ll_3       = log_loss(y3, p3, labels=["out", "walk", "hit"])
        base3      = np.array([(y3 == c).mean() for c in ["out", "walk", "hit"]])
        ll_3_naive = log_loss(y3, np.tile(base3, (len(y3), 1)), labels=["out", "walk", "hit"])

        y3_train        = np.where(
            train_df["pa_outcome"].isin({"single", "xbh", "home_run"}), "hit",
            np.where(train_df["pa_outcome"].isin({"strikeout", "out_in_play"}), "out", "walk")
        )
        naive3_sit_proba = _sit_naive_proba(train_df, test_df, ["out", "walk", "hit"],
                                             y_train=y3_train)
        ll_3_naive_sit   = log_loss(y3, naive3_sit_proba, labels=["out", "walk", "hit"])

        auc_3    = roc_auc_score(
            pd.get_dummies(pd.Series(y3))[["out", "walk", "hit"]].values,
            p3, multi_class="ovr", average="macro"
        )
        mlflow.log_metric("auc_3class",          auc_3)
        mlflow.log_metric("ll_3class_lift",       ll_3_naive - ll_3)
        mlflow.log_metric("ll_3class_sit_lift",   ll_3_naive_sit - ll_3)
        print(f"\n  3-class (out/walk/hit): AUC(macro)={auc_3:.4f}")
        print(f"    log loss: {ll_3:.4f}")
        print(f"    vs overall-rate naive:   {ll_3_naive:.4f}  lift={ll_3_naive - ll_3:+.4f}")
        print(f"    vs situation naive:      {ll_3_naive_sit:.4f}  lift={ll_3_naive_sit - ll_3:+.4f}"
              "  ← correct comparison")

        # Ordering check: do batters with higher career wOBA get higher predicted wOBA?
        chk = test_df[["batter_career_woba"]].copy()
        chk["exp_woba"]    = cascade["expected_woba"].values
        chk["actual_woba"] = actual_woba
        chk["pred_p_hr"]   = cascade["home_run"].values
        chk["pred_p_k"]    = cascade["strikeout"].values
        ordering = (
            chk
            .assign(woba_q=pd.qcut(
                chk["batter_career_woba"].fillna(chk["batter_career_woba"].median()),
                q=5, duplicates="drop"
            ))
            .groupby("woba_q", observed=True)
            .agg(n=("actual_woba", "count"),
                 actual_woba=("actual_woba", "mean"),
                 pred_ewoba=("exp_woba", "mean"),
                 pred_p_hr=("pred_p_hr", "mean"),
                 pred_p_k=("pred_p_k", "mean"))
            .round(3)
        )
        print("\n  Cascade distribution check by batter career wOBA quintile:")
        print(ordering.to_string())

        # ── Count state validation ────────────────────────────────────────────
        # Key sanity check for within-AB features: P(K) should rise with strikes,
        # P(walk) should rise with balls, monotonically across all count states.
        count_chk = test_df[["balls", "strikes"]].copy()
        count_chk["pred_k"]      = cascade["strikeout"].values
        count_chk["pred_walk"]   = cascade["walk"].values
        count_chk["pred_hr"]     = cascade["home_run"].values
        count_chk["actual_k"]    = (test_df["pa_outcome"] == "strikeout").astype(int).values
        count_chk["actual_walk"] = (test_df["pa_outcome"] == "walk").astype(int).values
        count_table = (
            count_chk
            .groupby(["balls", "strikes"])
            .agg(
                n=("pred_k", "count"),
                actual_k=("actual_k", "mean"),
                pred_k=("pred_k", "mean"),
                actual_walk=("actual_walk", "mean"),
                pred_walk=("pred_walk", "mean"),
                pred_hr=("pred_hr", "mean"),
            )
            .round(3)
        )
        print("\n  Count state validation (P(K) should rise with strikes, P(walk) with balls):")
        print(count_table.to_string())

        # ── Variance decomposition: career average vs situational signal ──────
        # ICC = var_between / (var_between + var_within)
        # ICC → 1.0: model is just predicting each batter's career average
        # ICC → 0.0: model is dominated by situational/pitcher variance within batters
        chk2 = test_df[["batter"]].copy()
        chk2["exp_woba"] = cascade["expected_woba"].values
        batter_means = chk2.groupby("batter")["exp_woba"].mean()
        var_between  = batter_means.var()
        var_within   = chk2.groupby("batter")["exp_woba"].var().mean()
        icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0
        print(f"\n  Variance decomposition of predicted E[wOBA]:")
        print(f"    Between-batter variance:  {var_between:.6f}")
        print(f"    Within-batter variance:   {var_within:.6f}  (situational/pitcher/count signal)")
        print(f"    ICC (career-avg fraction): {icc:.3f}  "
              f"({'model is mostly predicting career averages' if icc > 0.8 else 'meaningful situational signal present'})")

        # Platoon effect check: split by batter hand so LHB and RHB effects
        # don't cancel each other. For RHBs, diff = pred_vs_RHP - pred_vs_LHP
        # should be negative (RHBs hit LHP better). For LHBs it should be positive.
        chk3 = test_df[["batter", "stand", "p_throws"]].copy()
        chk3["exp_woba"] = cascade["expected_woba"].values
        platoon = (
            chk3.groupby(["batter", "stand", "p_throws"])["exp_woba"]
            .mean()
            .reset_index()
            .pivot_table(index=["batter", "stand"], columns="p_throws",
                         values="exp_woba")
            .dropna()
            .reset_index()
        )
        if "L" in platoon.columns and "R" in platoon.columns:
            platoon["diff"] = platoon["R"] - platoon["L"]
            print(f"\n  Platoon effect (pred E[wOBA] vs RHP minus vs LHP), by batter hand:")
            for hand, expected_sign in [("R", "negative — RHBs should hit LHP better"),
                                        ("L", "positive — LHBs should hit RHP better")]:
                grp = platoon[platoon["stand"] == hand]["diff"]
                if len(grp) > 0:
                    print(f"    {hand}HB (n={len(grp):,}): mean={grp.mean():.4f}  "
                          f"std={grp.std():.4f}  "
                          f"correct_direction={((grp < 0) if hand == 'R' else (grp > 0)).mean():.1%}"
                          f"  [{expected_sign}]")


def train_direct_classifiers(model_type, X_train, train_df, X_test, test_df):
    """Direct classifiers for binary and 3-class outcome prediction.

    Trained end-to-end on the simplified targets rather than reading off the
    cascade — gives a cleaner gradient signal for these specific bets.

    Uses the same 85/15 fit/calibration split as the cascade. No balanced class
    weights (they inflate minority-class probabilities); isotonic calibration on
    the holdout corrects residual miscalibration instead.

    Models:
      1. on-base vs out  (binary)
      2. hit / walk / out (3-class)
    """
    ON_BASE = {"single", "xbh", "home_run", "walk"}
    HIT     = {"single", "xbh", "home_run"}
    CLASSES = ["out", "walk", "hit"]

    # 85/15 fit / calibration split — mirrors the cascade
    cal_cut    = int(len(X_train) * 0.85)
    X_fit      = X_train.iloc[:cal_cut]
    X_cal      = X_train.iloc[cal_cut:]
    df_fit     = train_df.iloc[:cal_cut]
    df_cal     = train_df.iloc[cal_cut:]

    def _ob(df):
        return df["pa_outcome"].isin(ON_BASE).astype(int).values

    def _y3(df):
        return np.where(df["pa_outcome"].isin(HIT), "hit",
               np.where(df["pa_outcome"] == "walk",  "walk", "out"))

    y_ob_fit = _ob(df_fit);  y_ob_cal = _ob(df_cal);  y_ob_te = _ob(test_df)
    y3_fit   = _y3(df_fit);  y3_cal   = _y3(df_cal);  y3_te   = _y3(test_df)
    # full-train labels needed for situation naive baseline
    y_ob_tr  = _ob(train_df);  y3_tr = _y3(train_df)

    print(f"\n{'='*50}")
    print(f"Direct classifiers — {model_type.upper()}")
    print(f"{'='*50}")

    # ── Binary: on-base vs out ────────────────────────────────────────────────
    # No balanced weights — isotonic calibration on holdout corrects the offset.
    print("\nOn-base vs out...")
    ob_model = train_binary_stage(
        f"direct_onbase_{model_type}", X_fit, y_ob_fit, model_type, class_weight=None
    )
    raw_ob_cal = ob_model.predict_proba(X_cal)[:, 1]
    ob_cal     = IsotonicRegression(out_of_bounds="clip")
    ob_cal.fit(raw_ob_cal, y_ob_cal)

    p_ob = ob_cal.transform(ob_model.predict_proba(X_test)[:, 1])

    br_ob  = y_ob_te.mean()
    auc_ob = roc_auc_score(y_ob_te, p_ob)
    bs_ob  = brier_score_loss(y_ob_te, p_ob)
    bss_ob = 1.0 - bs_ob / brier_score_loss(y_ob_te, np.full(len(y_ob_te), br_ob))
    ll_ob  = log_loss(y_ob_te, np.column_stack([1 - p_ob, p_ob]))

    # Situation naive for binary: P(on-base | count, outs, base state)
    y_ob_tr_str    = np.where(y_ob_tr, "on_base", "out")
    y_ob_tr_str_tr = np.where(y_ob_tr, "on_base", "out")   # same, full train
    naive_ob_sit   = _sit_naive_proba(train_df, test_df, ["out", "on_base"],
                                       y_train=y_ob_tr_str)
    p_ob_sit_naive = naive_ob_sit[:, 1]   # P(on_base) column
    ll_ob_sit      = log_loss(y_ob_te,
                               np.column_stack([1 - p_ob_sit_naive, p_ob_sit_naive]))

    raw_mean = ob_model.predict_proba(X_cal)[:, 1].mean()
    cal_mean = p_ob.mean()
    print(f"  calibration holdout — raw: {raw_mean:.3f}  calibrated: {cal_mean:.3f}  "
          f"actual: {y_ob_cal.mean():.3f}")
    print(f"  Base rate: {br_ob:.3f}  AUC: {auc_ob:.4f}  BSS: {bss_ob:.4f}")
    print(f"  Log loss: {ll_ob:.4f}  vs situation naive: {ll_ob_sit:.4f}  "
          f"lift={ll_ob_sit - ll_ob:+.4f}")

    mlflow.log_metric(f"direct_ob_auc_{model_type}", auc_ob)
    mlflow.log_metric(f"direct_ob_bss_{model_type}", bss_ob)

    # ── 3-class: out / walk / hit ─────────────────────────────────────────────
    # No balanced weights — per-class isotonic calibration on holdout.
    print("\nHit / Walk / Out (3-class)...")
    m3, le3 = train_multiclass_stage(
        f"direct_3class_{model_type}", X_fit, y3_fit, CLASSES, model_type, sample_weight=None
    )

    raw_cal_p3 = m3.predict_proba(X_cal)   # shape (n_cal, 3) in le3.classes_ order
    col_idx    = [list(le3.classes_).index(c) for c in CLASSES]
    raw_cal_p3 = raw_cal_p3[:, col_idx]    # reorder to CLASSES

    # Per-class isotonic calibrators
    cal3 = []
    for i, c in enumerate(CLASSES):
        y_bin_cal = (y3_cal == c).astype(int)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_cal_p3[:, i], y_bin_cal)
        cal3.append(iso)

    raw_te_p3 = m3.predict_proba(X_test)[:, col_idx]
    p3 = np.column_stack([cal3[i].transform(raw_te_p3[:, i]) for i in range(3)])
    p3 = p3 / p3.sum(axis=1, keepdims=True)

    print(f"\n  {'Class':<8} {'Base rate':>10} {'AUC':>8} {'BSS':>8}")
    print(f"  {'-'*38}")
    for i, c in enumerate(CLASSES):
        y_bin = (y3_te == c).astype(int)
        br_c  = y_bin.mean()
        auc_c = roc_auc_score(y_bin, p3[:, i])
        bs_c  = brier_score_loss(y_bin, p3[:, i])
        bss_c = 1.0 - bs_c / brier_score_loss(y_bin, np.full(len(y_bin), br_c))
        print(f"  {c:<8} {br_c:>10.3f} {auc_c:>8.4f} {bss_c:>8.4f}")
        mlflow.log_metric(f"direct_{c}_auc_{model_type}", auc_c)
        mlflow.log_metric(f"direct_{c}_bss_{model_type}", bss_c)

    auc_3_macro = roc_auc_score(
        pd.get_dummies(pd.Series(y3_te))[CLASSES].values,
        p3, multi_class="ovr", average="macro"
    )

    ll_3           = log_loss(y3_te, p3, labels=CLASSES)
    ll_3_naive     = log_loss(y3_te,
                              np.tile([(y3_te == c).mean() for c in CLASSES], (len(y3_te), 1)),
                              labels=CLASSES)
    naive3_sit     = _sit_naive_proba(train_df, test_df, CLASSES, y_train=y3_tr)
    ll_3_naive_sit = log_loss(y3_te, naive3_sit, labels=CLASSES)

    print(f"\n  AUC (macro): {auc_3_macro:.4f}")
    print(f"  Log loss: {ll_3:.4f}")
    print(f"    vs overall-rate naive:   {ll_3_naive:.4f}  lift={ll_3_naive - ll_3:+.4f}")
    print(f"    vs situation naive:      {ll_3_naive_sit:.4f}  lift={ll_3_naive_sit - ll_3:+.4f}"
          "  ← correct comparison")
    mlflow.log_metric(f"direct_3class_auc_{model_type}",       auc_3_macro)
    mlflow.log_metric(f"direct_3class_sit_lift_{model_type}",  ll_3_naive_sit - ll_3)

    # ── Count state validation ────────────────────────────────────────────────
    chk = test_df[["balls", "strikes"]].copy()
    chk["pred_onbase"]   = p_ob
    chk["actual_onbase"] = y_ob_te
    chk["pred_hit"]      = p3[:, CLASSES.index("hit")]
    chk["actual_hit"]    = (y3_te == "hit").astype(int)
    chk["pred_walk"]     = p3[:, CLASSES.index("walk")]
    chk["actual_walk"]   = (y3_te == "walk").astype(int)
    count_chk = (
        chk.groupby(["balls", "strikes"])
        .agg(n=("pred_onbase", "count"),
             actual_onbase=("actual_onbase", "mean"),
             pred_onbase=("pred_onbase", "mean"),
             actual_hit=("actual_hit", "mean"),
             pred_hit=("pred_hit", "mean"),
             actual_walk=("actual_walk", "mean"),
             pred_walk=("pred_walk", "mean"))
        .round(3)
    )
    print("\n  Count state validation:")
    print(count_chk.to_string())

    # ── Save ─────────────────────────────────────────────────────────────────
    save_dir = MODEL_DIR / model_type
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ob_model, save_dir / "direct_onbase.pkl")
    joblib.dump(ob_cal,   save_dir / "direct_onbase_calibrator.pkl")
    joblib.dump(m3,       save_dir / "direct_3class.pkl")
    joblib.dump(le3,      save_dir / "direct_3class_le.pkl")
    joblib.dump(cal3,     save_dir / "direct_3class_calibrators.pkl")
    print(f"\n  Saved to {save_dir}/")


def main():
    print("Loading mid-AB features...")
    df = load_model2_midab_features()
    df["pa_outcome"] = df["pa_outcome"].replace({"triple": "xbh", "double": "xbh", "hbp": "walk"})
    df = build_pa_labels(df)

    df["same_hand"] = (df["stand"] == df["p_throws"]).astype(int)
    df["batter_platoon_woba_split"] = df["batter_career_woba_vs_rhp"] - df["batter_career_woba_vs_lhp"]
    df["pitcher_platoon_woba_split"] = (
        df["pitcher_career_woba_against_vs_rhb"] - df["pitcher_career_woba_against_vs_lhb"]
    )

    # BABIP luck residual: actual career BABIP minus the BABIP you'd expect from
    # the batter's batted-ball mix (GB ~0.24, LD ~0.68). Positive = outperforming
    # contact quality; negative = underperforming. Separates skill from variance.
    df["batter_career_babip_luck"] = (
        df["batter_career_babip"].astype(float)
        - (0.24 * df["batter_career_gb_rate"].astype(float)
           + 0.68 * df["batter_career_ld_rate"].astype(float))
    )

    unique_pas = df.groupby(["game_pk", "at_bat_number"]).ngroups
    print(f"Total pitch rows: {len(df):,}  |  Unique PAs: {unique_pas:,}")
    print(f"Avg pitches per PA: {len(df) / unique_pas:.2f}")
    print(f"In-play rate (pitch rows): {df['is_in_play'].mean():.3f}")

    train_df, test_df, cutoff_date = time_based_split(df)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,} | cutoff: {cutoff_date.date()}")

    X_train = prepare_features(train_df)
    X_test  = prepare_features(test_df)
    print(f"Features: {X_train.shape[1]}")

    mlflow.set_experiment("model2_midab_cascade")

    with mlflow.start_run(run_name="midab_cascade_comparison"):
        for mt in ["lgb", "xgb"]:
            train_and_evaluate(mt, X_train, train_df, X_test, test_df)
            train_direct_classifiers(mt, X_train, train_df, X_test, test_df)


if __name__ == "__main__":
    main()
