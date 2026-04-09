"""
main.py - BTC/USD Two-Stage Prediction Pipeline (LightGBM only)
=================================================
Two modes of operation:

  python main.py --full      Run everything including FinBERT (first time)
  python main.py             Skip FinBERT, use cached sentiment, just train & evaluate

Trains LightGBM as Model A and runs the full two-stage pipeline.

Architecture:
  Final_Prediction = Model_A(technical) + Model_B(sentiment_residuals)

For the simplest "sentiment vs no sentiment" comparison, run:  python compare_models.py
(A0 vs A1 in outputs/model_comparison.csv and outputs/ablation_summary.csv).
"""

import sys
import logging
import warnings
from pathlib import Path

import joblib
import pandas as pd

from config import (
    OUTPUT_NEWS_RAW, OUTPUT_NEWS_DAILY, OUTPUT_MERGED,
    OUTPUT_TRENDS_DAILY, OUTPUT_FGI_DAILY,
    MODEL_BUNDLE_PATH, TECHNICAL_FEATURES, SENTIMENT_FEATURES,
    LEGACY_NEWS_DAILY, LEGACY_TRENDS_DAILY, LEGACY_FGI_DAILY,
)

from step1_data       import (
    load_ohlcv, collect_all_news,
    fetch_google_trends_daily, fetch_fear_greed_daily,
)
from step2_features   import (add_technical_features, aggregate_daily_sentiment,
                               add_target, align_sentiment_to_ohlcv,
                               align_external_to_ohlcv,
                               merge_ohlcv_and_externals)
from step4_model_a    import train_model_a
from step5_model_b    import extract_residuals, train_model_b
from step6_evaluate   import evaluate_ensemble
from step7_backtest   import run_backtest

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def _ensure_parent_dir(path_str: str) -> None:
    p = Path(path_str)
    if p.parent and str(p.parent) not in (".", ""):
        p.parent.mkdir(parents=True, exist_ok=True)


# ===================================================================
# FULL mode: collect externals + run FinBERT + cache daily features
# ===================================================================
def run_full_and_cache_externals():
    """
    Run once (or when refreshing data). Caches:
      - news raw + FinBERT daily sentiment
      - Google Trends daily
      - Fear & Greed Index daily
    """
    from step3_sentiment import score_finbert

    print("=" * 60)
    print("STEP 1: Collecting News")
    print("=" * 60)

    df_news = collect_all_news()
    if df_news.empty:
        print("ERROR: No news data available.")
        return None
    _ensure_parent_dir(OUTPUT_NEWS_RAW)
    df_news.to_csv(OUTPUT_NEWS_RAW, index=False)
    log.info(f"Raw news saved -> {OUTPUT_NEWS_RAW}")

    print("\n" + "=" * 60)
    print("STEP 3: FinBERT Sentiment Scoring (one-time)")
    print("=" * 60)

    texts  = df_news["text"].tolist()
    scores = score_finbert(texts)
    df_news["label"]      = [s["label"]      for s in scores]
    df_news["confidence"] = [s["confidence"] for s in scores]
    df_news["numeric"]    = [s["numeric"]    for s in scores]

    df_daily = aggregate_daily_sentiment(df_news)
    _ensure_parent_dir(OUTPUT_NEWS_DAILY)
    df_daily.to_csv(OUTPUT_NEWS_DAILY, index=False)
    print(f"\n  Sentiment cached -> {OUTPUT_NEWS_DAILY}")
    print(f"  {len(df_daily)} days scored. You won't need to run FinBERT again.")

    print("\n" + "=" * 60)
    print("STEP 1B: Collecting Google Trends + Fear&Greed")
    print("=" * 60)

    try:
        df_trends = fetch_google_trends_daily()
        _ensure_parent_dir(OUTPUT_TRENDS_DAILY)
        df_trends.to_csv(OUTPUT_TRENDS_DAILY, index=False)
        print(f"  Trends cached -> {OUTPUT_TRENDS_DAILY} ({len(df_trends)} rows)")
    except Exception as e:
        print(f"  WARNING: Google Trends fetch failed: {e}")

    try:
        df_fgi = fetch_fear_greed_daily()
        _ensure_parent_dir(OUTPUT_FGI_DAILY)
        df_fgi.to_csv(OUTPUT_FGI_DAILY, index=False)
        print(f"  Fear&Greed cached -> {OUTPUT_FGI_DAILY} ({len(df_fgi)} rows)")
    except Exception as e:
        print(f"  WARNING: Fear&Greed fetch failed: {e}")

    return df_daily


# ===================================================================
# FAST mode: load cached sentiment CSV (no FinBERT needed)
# ===================================================================
def load_cached_sentiment():
    """Load previously scored sentiment from CSV."""
    path = Path(OUTPUT_NEWS_DAILY)
    legacy = Path(LEGACY_NEWS_DAILY)
    if not path.exists() and legacy.exists():
        path = legacy
    if not path.exists():
        print(f"ERROR: {OUTPUT_NEWS_DAILY} not found (or legacy {LEGACY_NEWS_DAILY}).")
        print("Run once with:  python main.py --full")
        return None

    df_daily = pd.read_csv(path)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    print(f"  Loaded cached sentiment: {len(df_daily)} days (from {path})")
    return df_daily


def _load_cached_external(path_str: str, date_col: str = "date") -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


# ===================================================================
# Run steps 5-6-7 for a single Model A variant
# ===================================================================
def run_pipeline_for_model(label, model_a, df_merged, tech_features,
                           sent_features, mask):
    """Run residual extraction, Model B, evaluation, and backtest."""
    print("\n" + "#" * 60)
    print(f"  PIPELINE: {label}")
    print("#" * 60)

    df_window = extract_residuals(model_a, df_merged, tech_features, mask)
    model_b, scaler_b, sent_used = train_model_b(df_window, sent_features)

    eval_results = evaluate_ensemble(
        df_merged, model_a, model_b, scaler_b,
        tech_features, sent_used, mask,
        model_label=label,
    )
    bt_results = run_backtest(eval_results, model_label=label)

    return {
        "model_a": model_a, "model_b": model_b,
        "scaler_b": scaler_b, "sent_used": sent_used,
        "eval": eval_results, "backtest": bt_results,
    }


# ===================================================================
# Main pipeline
# ===================================================================
def train_and_evaluate(df_daily):
    """Full pipeline: data -> features -> train both models -> compare."""

    # -- Step 1: Load OHLCV ------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Loading OHLCV Data")
    print("=" * 60)
    df_ohlcv = load_ohlcv()

    # -- Step 2: Technical features ----------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)
    df_ohlcv = add_technical_features(df_ohlcv)
    df_ohlcv = add_target(df_ohlcv)

    # -- Load cached external datasets -------------------------------------
    trends_path = OUTPUT_TRENDS_DAILY if Path(OUTPUT_TRENDS_DAILY).exists() else LEGACY_TRENDS_DAILY
    fgi_path    = OUTPUT_FGI_DAILY if Path(OUTPUT_FGI_DAILY).exists() else LEGACY_FGI_DAILY
    df_trends = _load_cached_external(trends_path, "date")
    df_fgi    = _load_cached_external(fgi_path, "date")

    # -- Align + merge externals (leakage-safe lagging happens in Step 2) ---
    df_sent_aligned   = align_sentiment_to_ohlcv(df_daily, df_ohlcv)
    df_trends_aligned = align_external_to_ohlcv(df_trends, df_ohlcv, prefix="trends_")
    df_fgi_aligned    = align_external_to_ohlcv(df_fgi, df_ohlcv, prefix="fgi_")

    df_merged = merge_ohlcv_and_externals(df_ohlcv, df_sent_aligned, df_trends_aligned, df_fgi_aligned)
    _ensure_parent_dir(OUTPUT_MERGED)
    df_merged.to_csv(OUTPUT_MERGED, index=False)

    print(f"\n  Final dataset: {len(df_merged):,} rows")
    print(f"  Date range:    {df_merged['Date'].min().date()} to "
          f"{df_merged['Date'].max().date()}")
    print(f"  Rows with news: {df_merged['has_news'].sum():,}")

    # -- Step 4: Train LightGBM (Model A) ----------------------------------
    tech_features = [f for f in TECHNICAL_FEATURES if f in df_merged.columns]
    sent_features = [f for f in SENTIMENT_FEATURES if f in df_merged.columns]

    model_a_results, mask = train_model_a(df_merged, tech_features)

    # -- Steps 5-6-7 -------------------------------------------------------
    model_a = model_a_results["lgbm"]["model"]
    pipeline = run_pipeline_for_model(
        "LightGBM", model_a, df_merged, tech_features, sent_features, mask
    )

    bundle = {
        "model_a":       pipeline["model_a"],
        "model_b":       pipeline["model_b"],
        "scaler_b":      pipeline["scaler_b"],
        "tech_features": tech_features,
        "sent_features": pipeline["sent_used"],
        "model_a_type":  "LightGBM",
    }
    _ensure_parent_dir(MODEL_BUNDLE_PATH)
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"\nModel saved -> {MODEL_BUNDLE_PATH}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# ===================================================================
# Entry point
# ===================================================================
def main():
    full_mode = "--full" in sys.argv

    if full_mode:
        print("MODE: Full pipeline (includes external data + FinBERT scoring)\n")
        df_daily = run_full_and_cache_externals()
    else:
        print("MODE: Fast pipeline (using cached sentiment)\n")
        df_daily = load_cached_sentiment()

    if df_daily is None:
        return

    train_and_evaluate(df_daily)


if __name__ == "__main__":
    main()
