"""
step2_features.py - Feature Engineering
========================================
Computes all features from the raw OHLCV and sentiment data:

  A) Technical indicators (from 5-year OHLCV):
     - Returns: log_return, return_lag1
     - Price action: hl_spread, oc_spread
     - Volume: vol_log, volume_change_pct, OBV, obv_change_pct
     - Moving averages: price_vs_sma{20,50,200}
     - Momentum: RSI-14, MACD histogram
     - Volatility: Bollinger %B, ATR%, rolling vol (10d, 30d)

  B) Sentiment aggregation (from FinBERT-scored news):
     - Daily: mean, std, max, min, count, pos/neg ratios
     - Rolling: 3-day momentum, 7-day momentum, bull/bear ratio

  C) Target construction:
     - target_next_log_return = tomorrow's log return (shift -1)
     - target_direction       = 1 if positive, 0 otherwise

  D) Merge OHLCV + sentiment on Date
"""

import logging

import numpy as np
import pandas as pd

from config import TARGET_COL, FEATURE_LAG_DAYS

log = logging.getLogger(__name__)


# ===========================================================================
# A) Technical Indicators
# ===========================================================================

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the OHLCV DataFrame.
    Expects columns: Date, Price, Open, High, Low, Volume.
    """
    # --- Returns -----------------------------------------------------------
    df["log_return"]  = np.log1p(df["Price"].pct_change())
    df["return_lag1"] = df["log_return"].shift(1)

    # --- Price action ------------------------------------------------------
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Price"]
    df["oc_spread"] = (df["Price"] - df["Open"]) / df["Open"]

    # --- Volume features ---------------------------------------------------
    df["vol_log"]           = np.log1p(df["Volume"])
    df["volume_change_pct"] = df["Volume"].pct_change()

    # On-Balance Volume (OBV): cumulative sum of signed volume
    sign = np.sign(df["Price"].diff()).fillna(0)
    df["obv"]            = (sign * df["Volume"]).cumsum()
    df["obv_change_pct"] = df["obv"].pct_change()

    # --- Moving averages vs price ------------------------------------------
    for window in [20, 50, 200]:
        sma = df["Price"].rolling(window).mean()
        df[f"price_vs_sma{window}"] = df["Price"] / sma - 1

    # --- RSI (14-day) ------------------------------------------------------
    delta = df["Price"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / (avg_l + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD histogram ----------------------------------------------------
    ema12 = df["Price"].ewm(span=12, adjust=False).mean()
    ema26 = df["Price"].ewm(span=26, adjust=False).mean()
    macd       = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal

    # --- Bollinger Bands %B ------------------------------------------------
    bb_mean  = df["Price"].rolling(20).mean()
    bb_std   = df["Price"].rolling(20).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    df["bband_pos"] = (df["Price"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # --- ATR % (Average True Range as percentage of price) -----------------
    high_low  = df["High"] - df["Low"]
    high_prev = (df["High"] - df["Price"].shift(1)).abs()
    low_prev  = (df["Low"]  - df["Price"].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr_pct"] = true_range.rolling(14).mean() / df["Price"]

    # --- Rolling volatility ------------------------------------------------
    df["vol_10d"] = df["log_return"].rolling(10).std()
    df["vol_30d"] = df["log_return"].rolling(30).std()

    # Guard against inf values from pct_change (e.g., when prior volume is 0)
    df = df.replace([np.inf, -np.inf], np.nan)

    log.info(f"  Technical features added: {16} indicators")
    return df


# ===========================================================================
# B) Sentiment Aggregation
# ===========================================================================

def aggregate_daily_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level FinBERT scores into one row per day.

    Input must have columns: date, label, numeric
    (produced by step3_sentiment.py)
    """
    df_news["date"] = pd.to_datetime(df_news["date"])

    daily = df_news.groupby("date").agg(
        sentiment_mean  = ("numeric", "mean"),
        sentiment_std   = ("numeric", lambda x: x.std(ddof=0)),
        sentiment_max   = ("numeric", "max"),
        sentiment_min   = ("numeric", "min"),
        sentiment_count = ("numeric", "count"),
        positive_ratio  = ("label", lambda x: (x == "positive").mean()),
        negative_ratio  = ("label", lambda x: (x == "negative").mean()),
    ).reset_index()

    daily["sentiment_range"] = daily["sentiment_max"] - daily["sentiment_min"]
    daily = daily.sort_values("date").reset_index(drop=True)

    # Rolling momentum: is sentiment getting better or worse?
    daily["sentiment_mom_3d"] = daily["sentiment_mean"].rolling(3, min_periods=1).mean()
    daily["sentiment_mom_7d"] = daily["sentiment_mean"].rolling(7, min_periods=1).mean()

    # Bull/bear divergence
    daily["bull_bear_ratio"] = (
        daily["positive_ratio"] / (daily["negative_ratio"] + 1e-6)
    ).clip(upper=10)

    log.info(f"  Sentiment aggregated: {len(daily)} unique days")
    return daily


# ===========================================================================
# C) Target Construction
# ===========================================================================

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the prediction target: next day's log return.
    Uses shift(-1) so row t contains tomorrow's return as the target.
    """
    df[TARGET_COL]         = df["log_return"].shift(-1)
    df["target_direction"] = (df[TARGET_COL] > 0).astype(int)
    return df


# ===========================================================================
# D) Merge & Align
# ===========================================================================

def align_sentiment_to_ohlcv(df_sentiment: pd.DataFrame,
                              df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex daily sentiment to match OHLCV trading dates.
    BTC trades 24/7, so most days align directly.
    Forward-fills any gaps (a day with no news inherits yesterday's sentiment).
    """
    trading_dates = pd.to_datetime(df_ohlcv["Date"]).sort_values().unique()
    date_index    = pd.DatetimeIndex(trading_dates)

    df_sentiment = df_sentiment.set_index("date").reindex(date_index)

    # Forward fill sentiment (not count -- no-news days should show 0 articles)
    ffill_cols = [c for c in df_sentiment.columns if c != "sentiment_count"]
    df_sentiment[ffill_cols] = df_sentiment[ffill_cols].ffill()
    df_sentiment["sentiment_count"] = df_sentiment["sentiment_count"].fillna(0)

    # Remaining NaN (start of series) -> neutral
    df_sentiment = df_sentiment.fillna(0.0)

    # Leakage control: shift sentiment features forward so day t features
    # reflect information available up to end of day t-1 (conservative).
    if FEATURE_LAG_DAYS and FEATURE_LAG_DAYS > 0:
        df_sentiment = df_sentiment.shift(FEATURE_LAG_DAYS)
        df_sentiment = df_sentiment.fillna(0.0)

    df_sentiment = df_sentiment.reset_index().rename(columns={"index": "date"})
    return df_sentiment


def align_external_to_ohlcv(df_external: pd.DataFrame,
                            df_ohlcv: pd.DataFrame,
                            date_col: str = "date",
                            prefix: str = "") -> pd.DataFrame:
    """
    Generic alignment for external daily time series (Trends / Fear&Greed).

    - Reindexes to OHLCV daily dates
    - Forward-fills missing values (external series typically persists day-to-day)
    - Applies leakage-safe lagging via FEATURE_LAG_DAYS

    Args:
        df_external: DataFrame with a date column + feature columns
        df_ohlcv:    OHLCV DataFrame with 'Date'
        date_col:    Name of date column in df_external
        prefix:      If provided, only columns starting with this prefix are treated
                    as feature columns (all others except date_col are dropped)
    """
    trading_dates = pd.to_datetime(df_ohlcv["Date"]).sort_values().unique()
    date_index = pd.DatetimeIndex(trading_dates)

    if df_external is None or df_external.empty:
        return pd.DataFrame({"date": date_index})

    df = df_external.copy()
    if date_col not in df.columns:
        return pd.DataFrame({"date": date_index})

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    feat_cols = [c for c in df.columns if c != date_col]
    if prefix:
        feat_cols = [c for c in feat_cols if c.startswith(prefix)]

    df = df[[date_col] + feat_cols].drop_duplicates(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col).reindex(date_index)

    if feat_cols:
        df[feat_cols] = df[feat_cols].ffill().fillna(0.0)

        if FEATURE_LAG_DAYS and FEATURE_LAG_DAYS > 0:
            df[feat_cols] = df[feat_cols].shift(FEATURE_LAG_DAYS).fillna(0.0)

    df = df.reset_index().rename(columns={"index": "date"})
    return df


def merge_ohlcv_and_sentiment(df_ohlcv: pd.DataFrame,
                               df_sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join OHLCV (with technical features) and daily sentiment on Date.
    Adds a 'has_news' flag for rows with real news data.
    """
    df_sentiment = df_sentiment.rename(columns={"date": "Date"})
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])

    merged = df_ohlcv.merge(df_sentiment, on="Date", how="left")

    # Fill sentiment NaN with neutral (days before news collection)
    sent_cols = [c for c in df_sentiment.columns if c != "Date"]
    merged[sent_cols] = merged[sent_cols].fillna(0.0)

    # Flag rows with real news
    merged["has_news"] = merged["sentiment_count"] > 0

    # Drop last row (target is NaN because there is no tomorrow)
    merged = merged.dropna(subset=[TARGET_COL])

    log.info(f"  Merged dataset: {len(merged):,} rows, "
             f"{merged['has_news'].sum():,} with news")
    return merged


def merge_ohlcv_and_externals(df_ohlcv: pd.DataFrame,
                              df_sentiment: pd.DataFrame,
                              df_trends: pd.DataFrame | None = None,
                              df_fgi: pd.DataFrame | None = None,
                              drop_incomplete_target: bool = True) -> pd.DataFrame:
    """
    Left-join OHLCV with:
      - daily sentiment features (FinBERT)
      - Google Trends features
      - Fear & Greed Index features

    If drop_incomplete_target is True (default), drops rows with no realized next-day
    target (training/eval). Set False to keep the final row for daily inference.
    """
    merged = df_ohlcv.copy()

    # Sentiment
    df_sent = df_sentiment.rename(columns={"date": "Date"})
    df_sent["Date"] = pd.to_datetime(df_sent["Date"])
    merged = merged.merge(df_sent, on="Date", how="left")

    # Trends
    if df_trends is not None and not df_trends.empty:
        df_t = df_trends.rename(columns={"date": "Date"})
        df_t["Date"] = pd.to_datetime(df_t["Date"])
        merged = merged.merge(df_t, on="Date", how="left")

    # Fear & Greed
    if df_fgi is not None and not df_fgi.empty:
        df_g = df_fgi.rename(columns={"date": "Date"})
        df_g["Date"] = pd.to_datetime(df_g["Date"])
        merged = merged.merge(df_g, on="Date", how="left")

    # Fill all external NaNs with 0.0 (conservative; avoids dropping rows)
    external_cols = [c for c in merged.columns if c not in ("Date", "Price", "Open", "High", "Low", "Volume", "Change_pct")]
    merged[external_cols] = merged[external_cols].fillna(0.0)

    # has_news flag (still based on *lagged* sentiment_count)
    if "sentiment_count" in merged.columns:
        merged["has_news"] = merged["sentiment_count"] > 0
    else:
        merged["has_news"] = False

    if drop_incomplete_target:
        merged = merged.dropna(subset=[TARGET_COL])

    log.info(
        f"  Merged dataset (externals): {len(merged):,} rows, "
        f"{int(merged['has_news'].sum()) if 'has_news' in merged.columns else 0:,} with news"
    )
    return merged
