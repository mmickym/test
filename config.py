"""
config.py - Central configuration for the BTC Two-Stage Prediction Model
=========================================================================
All paths, feature lists, hyperparameters, and constants live here.
Import this from every step file so nothing is hardcoded elsewhere.
"""

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
OHLCV_PATH         = "Bitcoin Historical Data.csv"
PRESCRAPED_NEWS    = "news_prescraped_seed.csv"

# Clean output structure (new files go into these folders)
CACHE_DIR          = "cache"
OUTPUTS_DIR        = "outputs"

# Cached datasets (created by `python main.py --full`)
OUTPUT_NEWS_RAW     = f"{CACHE_DIR}/btc_news_raw.csv"
OUTPUT_NEWS_DAILY   = f"{CACHE_DIR}/btc_news_daily_sentiment.csv"
OUTPUT_TRENDS_DAILY = f"{CACHE_DIR}/btc_google_trends_daily.csv"
OUTPUT_FGI_DAILY    = f"{CACHE_DIR}/btc_fear_greed_daily.csv"

# Final merged dataset + artifacts
OUTPUT_MERGED       = f"{OUTPUTS_DIR}/btc_merged_dataset.csv"
MODEL_BUNDLE_PATH   = f"{OUTPUTS_DIR}/btc_two_stage_bundle.pkl"
EVAL_CHART_PATH     = f"{OUTPUTS_DIR}/btc_two_stage_evaluation.png"
BACKTEST_CHART_PATH = f"{OUTPUTS_DIR}/btc_backtest_equity_curve.png"

# Legacy filenames (backwards compatible reads)
LEGACY_NEWS_RAW     = "btc_news_raw.csv"
LEGACY_NEWS_DAILY   = "btc_news_daily_sentiment.csv"
LEGACY_PRESCRAPED_NEWS = "bitcoin_65_days_news.csv"
LEGACY_TRENDS_DAILY = "btc_google_trends_daily.csv"
LEGACY_FGI_DAILY    = "btc_fear_greed_daily.csv"

# ---------------------------------------------------------------------------
# Backtesting Parameters
# ---------------------------------------------------------------------------
STOP_LOSS_PCT      = -0.02     # -2% stop loss
TAKE_PROFIT_PCT    =  0.04     # +4% take profit (risk:reward = 1:2)
TRADING_FEE_PCT    =  0.001    # 0.1% per trade (exchange fee)

# ---------------------------------------------------------------------------
# News Scraping
# ---------------------------------------------------------------------------
CRYPTOPANIC_TOKEN  = ""        # leave empty to skip
NEWSDATA_KEY       = ""        # leave empty to skip
DAYS_BACK          = 90        # RSS feeds typically keep 30-90 days

# FinBERT model for sentiment scoring
FINBERT_MODEL      = "ProsusAI/finbert"

# ---------------------------------------------------------------------------
# External datasets (time-joined)
# ---------------------------------------------------------------------------
# Conservative leakage control: shift external daily features by 1 day
FEATURE_LAG_DAYS   = 1

# Google Trends configuration (pytrends)
TRENDS_KEYWORDS = [
    "bitcoin",
    "btc",
    "bitcoin price",
]
TRENDS_GEO         = ""        # empty = worldwide
TRENDS_DAYS_BACK   = 450       # enough to cover evaluation windows + lags

# Fear & Greed Index configuration (Alternative.me)
FGI_LIMIT          = 0         # 0 = all available history

# ---------------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------------
NEWS_WINDOW_DAYS   = 65        # last N days reserved for Model B
TARGET_COL         = "target_next_log_return"

# ---------------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------------

# Technical features used by Model A (5-year baseline)
TECHNICAL_FEATURES = [
    # Returns & price action
    "log_return",
    "return_lag1",         # previous day's return (autoregressive signal)
    "hl_spread",           # intraday high-low range / price
    "oc_spread",           # open-to-close change / open

    # Volume
    "vol_log",             # log(1 + volume)
    "volume_change_pct",   # day-over-day volume % change
    "obv_change_pct",      # on-balance volume rate of change

    # Distance from moving averages
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma200",

    # Momentum / oscillators
    "rsi_14",
    "macd_hist",

    # Volatility
    "bband_pos",           # Bollinger Band %B (position within bands)
    "atr_pct",             # Average True Range as % of price
    "vol_10d",             # 10-day rolling std of log returns
    "vol_30d",             # 30-day rolling std of log returns
]

# Sentiment features used by Model B (65-day news corrector)
SENTIMENT_FEATURES = [
    "sentiment_mean",      # daily average FinBERT score (-1 to +1)
    "sentiment_std",       # intra-day disagreement
    "sentiment_max",       # most bullish headline
    "sentiment_min",       # most bearish headline
    "sentiment_count",     # number of articles
    "positive_ratio",      # fraction positive
    "negative_ratio",      # fraction negative
    "sentiment_range",     # max - min (news volatility)
    "sentiment_mom_3d",    # 3-day rolling mean (momentum)
    "sentiment_mom_7d",    # 7-day rolling mean
    "bull_bear_ratio",     # positive / negative ratio
]

# External (non-news) features
TRENDS_FEATURES = [
    "trends_bitcoin",
    "trends_btc",
    "trends_bitcoin_price",
    "trends_mean",
]

FGI_FEATURES = [
    "fgi_value",
    "fgi_value_norm",
]

# ---------------------------------------------------------------------------
# RSS Feeds (no API key needed)
# ---------------------------------------------------------------------------
RSS_FEEDS = {
    "coindesk":         "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph":    "https://cointelegraph.com/rss",
    "decrypt":          "https://decrypt.co/feed",
    "bitcoin_magazine": "https://bitcoinmagazine.com/.rss/full/",
    "cryptoslate":      "https://cryptoslate.com/feed/",
    "theblock":         "https://www.theblock.co/rss.xml",
}

# Keywords for filtering BTC-relevant articles
BTC_KEYWORDS = {
    "bitcoin", "btc", "crypto", "cryptocurrency", "blockchain",
    "satoshi", "lightning network", "halving", "coinbase", "binance",
    "sec", "etf", "tether", "usdt", "stablecoin", "defi",
}
