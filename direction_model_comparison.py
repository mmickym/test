"""
direction_model_comparison.py
=============================
Classification benchmark for next-day BTC direction (Up/Down).

Models:
  - Logistic Regression (baseline)
  - LightGBMClassifier (nonlinear tabular model)

Metrics (on a future time window, no shuffling):
  - Accuracy
  - Precision
  - Recall
  - F1

Run:
  python direction_model_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb

from config import (
    OUTPUT_NEWS_DAILY,
    LEGACY_NEWS_DAILY,
    OUTPUT_TRENDS_DAILY,
    LEGACY_TRENDS_DAILY,
    OUTPUT_FGI_DAILY,
    LEGACY_FGI_DAILY,
    OUTPUTS_DIR,
    NEWS_WINDOW_DAYS,
    TECHNICAL_FEATURES,
    SENTIMENT_FEATURES,
    TRENDS_FEATURES,
    FGI_FEATURES,
)

from step1_data import load_ohlcv
from step2_features import (
    add_technical_features,
    add_target,
    align_sentiment_to_ohlcv,
    align_external_to_ohlcv,
    merge_ohlcv_and_externals,
)


def _load_cached_daily(path: str, date_col: str = "date") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def _build_dataset() -> pd.DataFrame:
    df_ohlcv = load_ohlcv()
    df_ohlcv = add_technical_features(df_ohlcv)
    df_ohlcv = add_target(df_ohlcv)

    news_path = OUTPUT_NEWS_DAILY if Path(OUTPUT_NEWS_DAILY).exists() else LEGACY_NEWS_DAILY
    df_sent = _load_cached_daily(news_path, "date")
    if df_sent.empty:
        raise FileNotFoundError(
            f"Missing {OUTPUT_NEWS_DAILY} (or legacy {LEGACY_NEWS_DAILY}). Run `python main.py --full` once first."
        )

    trends_path = OUTPUT_TRENDS_DAILY if Path(OUTPUT_TRENDS_DAILY).exists() else LEGACY_TRENDS_DAILY
    fgi_path = OUTPUT_FGI_DAILY if Path(OUTPUT_FGI_DAILY).exists() else LEGACY_FGI_DAILY
    df_trends = _load_cached_daily(trends_path, "date")
    df_fgi = _load_cached_daily(fgi_path, "date")

    sent_aligned = align_sentiment_to_ohlcv(df_sent, df_ohlcv)
    trends_aligned = align_external_to_ohlcv(df_trends, df_ohlcv, prefix="trends_")
    fgi_aligned = align_external_to_ohlcv(df_fgi, df_ohlcv, prefix="fgi_")

    df = merge_ohlcv_and_externals(df_ohlcv, sent_aligned, trends_aligned, fgi_aligned)
    return df


def _feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    tech = [c for c in TECHNICAL_FEATURES if c in df.columns]
    sent = [c for c in SENTIMENT_FEATURES if c in df.columns]
    trends = [c for c in TRENDS_FEATURES if c in df.columns]
    fgi = [c for c in FGI_FEATURES if c in df.columns]

    return {
        "C0_tech_only": tech,
        "C1_tech_plus_sentiment": tech + sent,
        "C2_tech_plus_all": tech + sent + trends + fgi,
    }


def _time_split_last_window(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= window + 50:
        raise ValueError("Not enough rows for a stable time split.")
    train = df.iloc[:-window].copy()
    test = df.iloc[-window:].copy()
    return train, test


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    print("=" * 80)
    print("BTC DIRECTION CLASSIFICATION — MODEL COMPARISON (ENGLISH)")
    print("=" * 80)

    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    df = _build_dataset()

    if "target_direction" not in df.columns:
        raise KeyError("target_direction not found in dataset.")

    # Drop rows with missing target
    df = df.dropna(subset=["target_direction"]).reset_index(drop=True)

    train_df, test_df = _time_split_last_window(df, NEWS_WINDOW_DAYS)
    y_train = train_df["target_direction"].astype(int).values
    y_test = test_df["target_direction"].astype(int).values

    feat_sets = _feature_sets(df)
    rows: list[dict] = []

    for variant, feats in feat_sets.items():
        feats = [f for f in feats if f in df.columns]
        X_train_df = train_df[feats].fillna(0.0)
        X_test_df = test_df[feats].fillna(0.0)
        X_train = X_train_df.values
        X_test = X_test_df.values

        # Logistic Regression baseline (scaled)
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ])
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        m_lr = _eval_metrics(y_test, pred_lr)
        rows.append({
            "variant": variant,
            "model": "LogisticRegression",
            "n_features": len(feats),
            **m_lr,
        })

        # LightGBM classifier
        lgbm = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        # Fit/predict using DataFrames to avoid "invalid feature names" warnings
        lgbm.fit(X_train_df, y_train)
        pred_lgbm = lgbm.predict(X_test_df)
        m_lgbm = _eval_metrics(y_test, pred_lgbm)
        rows.append({
            "variant": variant,
            "model": "LightGBMClassifier",
            "n_features": len(feats),
            **m_lgbm,
        })

        # Print confusion matrix for the best of the two on this variant (by F1)
        best_pred = pred_lgbm if m_lgbm["f1"] >= m_lr["f1"] else pred_lr
        cm = confusion_matrix(y_test, best_pred)
        print(f"\n{variant} — confusion matrix (best model in variant):")
        print(cm)

    out = pd.DataFrame(rows).sort_values(["variant", "f1", "accuracy"], ascending=[True, False, False]).reset_index(drop=True)
    out_path = Path(OUTPUTS_DIR) / "direction_model_comparison.csv"
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("TOP RESULTS (sorted by F1 desc)")
    print("=" * 80)
    print(out.sort_values(["f1", "accuracy"], ascending=[False, False]).head(10).to_string(index=False))
    print(f"\nSaved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()

