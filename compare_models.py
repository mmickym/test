"""
compare_models.py - Ablation runner + model comparison
======================================================
Evaluates multiple feature sets (ablations) for predicting next-day BTC log returns
and outputs `outputs/model_comparison.csv` suitable for presentation.

We compare:
  - LightGBM only (Model A family)
  - A0..A4 single-stage variants
  - Two-stage (existing pipeline: Model A tech + Model B residual corrector)

Run:
  python compare_models.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    OUTPUT_NEWS_DAILY,
    OUTPUT_TRENDS_DAILY,
    OUTPUT_FGI_DAILY,
    LEGACY_NEWS_DAILY,
    LEGACY_TRENDS_DAILY,
    LEGACY_FGI_DAILY,
    NEWS_WINDOW_DAYS,
    TARGET_COL,
    TECHNICAL_FEATURES,
    SENTIMENT_FEATURES,
    TRENDS_FEATURES,
    FGI_FEATURES,
    OUTPUTS_DIR,
)

from step1_data import load_ohlcv
from step2_features import (
    add_technical_features,
    add_target,
    align_sentiment_to_ohlcv,
    align_external_to_ohlcv,
    merge_ohlcv_and_externals,
)
from step4_model_a import train_model_a
from step5_model_b import extract_residuals, train_model_b
from step6_evaluate import evaluate_model_a_only, evaluate_ensemble
from step7_backtest import run_backtest


warnings.filterwarnings("ignore", category=FutureWarning)


def _load_cached_daily(path: str, date_col: str = "date") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def _build_master_dataset() -> pd.DataFrame:
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
        "A0_tech_only": tech,
        "A1_tech_plus_sentiment": tech + sent,
        "A2_tech_plus_trends": tech + trends,
        "A3_tech_plus_fgi": tech + fgi,
        "A4_tech_plus_all": tech + sent + trends + fgi,
    }


def main():
    print("=" * 70)
    print("BTC/USD Model Comparison (Ablations + Two-stage, LightGBM only)")
    print("=" * 70)

    df = _build_master_dataset()
    feat_sets = _feature_sets(df)

    rows: list[dict] = []

    # Use the same evaluation window everywhere (last NEWS_WINDOW_DAYS rows after NaN mask)
    for variant_name, features in feat_sets.items():
        print("\n" + "-" * 70)
        print(f"VARIANT: {variant_name}")
        print("-" * 70)

        model_a_results, mask = train_model_a(df, tech_features=features, news_window=NEWS_WINDOW_DAYS)
        model_a = model_a_results["lgbm"]["model"]

        eval_results = evaluate_model_a_only(
            df=df,
            model_a=model_a,
            features=features,
            mask=mask,
            model_label=f"{variant_name}_LightGBM",
        )
        bt_results = run_backtest(eval_results, model_label=f"{variant_name}_LightGBM")

        rows.append({
            "variant": variant_name,
            "model": "LightGBM",
            "n_features": len(features),
            "rmse": eval_results["rmse_a"],
            "dir_acc": eval_results["dir_acc_a"],
            "backtest_return_pct": bt_results[0]["total_return"],  # Model A strategy
            "backtest_sharpe": bt_results[0]["sharpe"],
            "backtest_max_dd": bt_results[0]["max_drawdown"],
        })

    # Two-stage evaluation (keeps original design: Model A uses technical-only)
    print("\n" + "-" * 70)
    print("VARIANT: Two-stage (tech Model A + residual Model B)")
    print("-" * 70)
    tech_only = [c for c in TECHNICAL_FEATURES if c in df.columns]
    sent_feats = [c for c in SENTIMENT_FEATURES if c in df.columns]

    model_a_results, mask = train_model_a(df, tech_features=tech_only, news_window=NEWS_WINDOW_DAYS)
    model_a = model_a_results["lgbm"]["model"]
    df_window = extract_residuals(model_a, df, tech_only, mask, news_window=NEWS_WINDOW_DAYS)
    model_b, scaler_b, sent_used = train_model_b(df_window, sent_feats)

    eval_results = evaluate_ensemble(
        df=df,
        model_a=model_a,
        model_b=model_b,
        scaler_b=scaler_b,
        tech_features=tech_only,
        sent_features=sent_used,
        mask=mask,
        model_label="TwoStage_LightGBM",
    )
    bt_results = run_backtest(eval_results, model_label="TwoStage_LightGBM")

    rows.append({
        "variant": "TwoStage",
        "model": "LightGBM",
        "n_features": len(tech_only) + len(sent_used),
        "rmse": eval_results["rmse_ensemble"],
        "dir_acc": eval_results["dir_acc_ensemble"],
        "backtest_return_pct": bt_results[1]["total_return"],  # ensemble strategy
        "backtest_sharpe": bt_results[1]["sharpe"],
        "backtest_max_dd": bt_results[1]["max_drawdown"],
    })

    out = pd.DataFrame(rows).sort_values(["variant", "model"]).reset_index(drop=True)
    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(OUTPUTS_DIR) / "model_comparison.csv"
    out.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # Ablation: does sentiment help? (A1 vs A0)
    # ------------------------------------------------------------------
    ablation_path = Path(OUTPUTS_DIR) / "ablation_summary.csv"
    try:
        a0 = out[out["variant"] == "A0_tech_only"].iloc[0]
        a1 = out[out["variant"] == "A1_tech_plus_sentiment"].iloc[0]
        dir_acc_lift_pp = (float(a1["dir_acc"]) - float(a0["dir_acc"])) * 100.0
        rmse_change_pct = (float(a1["rmse"]) - float(a0["rmse"])) / float(a0["rmse"]) * 100.0
        df_ab = pd.DataFrame([{
            "baseline_variant": "A0_tech_only",
            "sentiment_variant": "A1_tech_plus_sentiment",
            "dir_acc_baseline": float(a0["dir_acc"]),
            "dir_acc_with_sentiment": float(a1["dir_acc"]),
            "dir_acc_lift_pp": dir_acc_lift_pp,
            "rmse_baseline": float(a0["rmse"]),
            "rmse_with_sentiment": float(a1["rmse"]),
            "rmse_change_pct": rmse_change_pct,
        }])
        df_ab.to_csv(ablation_path, index=False)
        print("\n" + "=" * 70)
        print("SENTIMENT ABLATION (A1 vs A0)")
        print("=" * 70)
        print(f"- Directional accuracy lift: {dir_acc_lift_pp:+.2f} percentage points")
        print(f"- RMSE change: {rmse_change_pct:+.2f}% (negative is better)")
        print(f"- Saved -> {ablation_path.resolve()}")
    except Exception as e:
        print(f"\nWARNING: Could not compute sentiment ablation summary: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by dir_acc desc, then rmse asc)")
    print("=" * 70)
    print(out.sort_values(["dir_acc", "rmse"], ascending=[False, True]).head(10).to_string(index=False))
    print(f"\nSaved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()

