"""
step1_data.py - Data Loading & Cleaning
========================================
Loads two raw data sources:
  1. Bitcoin Historical Data (5 years of OHLCV from Investing.com)
  2. News articles (pre-scraped CSV or live RSS/API scraping)

Outputs clean DataFrames ready for feature engineering in Step 2.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import pandas as pd
import requests

from config import (
    OHLCV_PATH, PRESCRAPED_NEWS, LEGACY_PRESCRAPED_NEWS, CRYPTOPANIC_TOKEN, NEWSDATA_KEY,
    DAYS_BACK, RSS_FEEDS, BTC_KEYWORDS,
    TRENDS_KEYWORDS, TRENDS_GEO, TRENDS_DAYS_BACK,
    FGI_LIMIT,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: BTC relevance filter
# ---------------------------------------------------------------------------
def is_btc_relevant(text: str) -> bool:
    """Return True if the text mentions any BTC-related keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in BTC_KEYWORDS)


# ---------------------------------------------------------------------------
# 1A. Load OHLCV (Investing.com format)
# ---------------------------------------------------------------------------
def load_ohlcv(path: str = OHLCV_PATH) -> pd.DataFrame:
    """
    Parse the Investing.com BTC/USD CSV into a clean DataFrame.

    Handles:
      - Comma-separated price strings ("71,108.5" -> 71108.5)
      - Volume suffixes ("68.29K" -> 68290.0)
      - Missing volume entries ("-" -> 0.0)
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Clean price columns: remove commas, cast to float
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = df[col].str.replace(",", "").astype(float)

    # Parse volume strings (e.g. "68.29K", "1.2M", "-")
    def parse_vol(v):
        v = str(v).strip()
        if v in ("-", "", "nan"):
            return 0.0
        multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
        if v[-1] in multipliers:
            return float(v[:-1]) * multipliers[v[-1]]
        return float(v)

    df["Volume"]     = df["Vol."].apply(parse_vol)
    df["Change_pct"] = df["Change %"].str.replace("%", "").astype(float)

    # Drop original messy columns, sort chronologically
    df = df.drop(columns=["Vol.", "Change %"])
    df = df.sort_values("Date").reset_index(drop=True)

    log.info(f"OHLCV loaded: {len(df)} rows "
             f"({df['Date'].min().date()} to {df['Date'].max().date()})")
    return df


# ---------------------------------------------------------------------------
# 1B. Load pre-scraped news CSV
# ---------------------------------------------------------------------------
def load_prescraped_news(path: str = PRESCRAPED_NEWS) -> pd.DataFrame:
    """
    Load a pre-scraped news CSV (e.g. news_prescraped_seed.csv).
    Normalizes column names to the standard format: date, source, title, summary, text.
    """
    resolved = path
    if not resolved or not Path(resolved).exists():
        if path == PRESCRAPED_NEWS and LEGACY_PRESCRAPED_NEWS and Path(LEGACY_PRESCRAPED_NEWS).exists():
            resolved = LEGACY_PRESCRAPED_NEWS
        else:
            return pd.DataFrame()
    path = resolved

    log.info(f"Loading pre-scraped news: {path}")
    df = pd.read_csv(path)

    # Normalize column names to a common schema
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("publishedat", "published_at", "pub_date"):
            col_map[col] = "publishedAt"
        elif cl == "title":
            col_map[col] = "title"
        elif cl in ("description", "summary", "content"):
            col_map[col] = "description"
    df = df.rename(columns=col_map)

    if "publishedAt" not in df.columns or "title" not in df.columns:
        log.warning("Pre-scraped CSV missing required columns. Skipping.")
        return pd.DataFrame()

    df["date"]    = pd.to_datetime(df["publishedAt"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["source"]  = "prescraped"
    df["title"]   = df["title"].fillna("")
    df["summary"] = df.get("description", pd.Series([""] * len(df))).fillna("").str[:500]
    df["text"]    = (df["title"] + ". " + df["summary"]).str[:512]

    # Keep only BTC-relevant articles
    df = df[df["text"].apply(is_btc_relevant)].reset_index(drop=True)
    log.info(f"  BTC-relevant articles: {len(df):,}")

    return df[["date", "source", "title", "summary", "text"]]


# ---------------------------------------------------------------------------
# 1C. Scrape RSS feeds (live)
# ---------------------------------------------------------------------------
def scrape_rss(cutoff_date: datetime) -> pd.DataFrame:
    """Scrape all configured RSS feeds. Returns raw article DataFrame."""
    records = []

    for source, url in RSS_FEEDS.items():
        log.info(f"  RSS: {source}")
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; BTCResearchBot/1.0)"}
            resp = requests.get(url, headers=headers, timeout=15)
            feed = feedparser.parse(resp.text)

            for entry in feed.entries:
                published = None
                for field in ("published_parsed", "updated_parsed", "created_parsed"):
                    if hasattr(entry, field) and getattr(entry, field):
                        t = getattr(entry, field)
                        published = datetime(*t[:6])
                        break
                if published is None or published < cutoff_date:
                    continue

                title   = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or ""
                text    = f"{title}. {summary}".strip()

                if not is_btc_relevant(text):
                    continue

                records.append({
                    "date":    published.date(),
                    "source":  source,
                    "title":   title,
                    "summary": summary[:500],
                    "text":    text[:512],
                })
        except Exception as e:
            log.warning(f"  Failed {source}: {e}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 1D. CryptoPanic API (optional)
# ---------------------------------------------------------------------------
def scrape_cryptopanic(token: str, days_back: int) -> pd.DataFrame:
    """Free tier: 50 req/day, 20 articles per page."""
    if not token:
        return pd.DataFrame()

    log.info("  CryptoPanic API...")
    records = []
    cutoff = datetime.now() - timedelta(days=days_back)
    url = "https://cryptopanic.com/api/v1/posts/"
    page = 1

    while True:
        params = {
            "auth_token": token, "currencies": "BTC",
            "filter": "news", "public": "true", "page": page,
        }
        try:
            data = requests.get(url, params=params, timeout=10).json()
        except Exception as e:
            log.warning(f"  CryptoPanic page {page} failed: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            try:
                pub = datetime.fromisoformat(
                    item.get("published_at", "").replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except Exception:
                continue
            if pub < cutoff:
                return pd.DataFrame(records)

            title = item.get("title", "")
            records.append({
                "date": pub.date(), "source": "cryptopanic",
                "title": title, "summary": "", "text": title,
            })

        page += 1
        time.sleep(1.2)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 1E. NewsData.io API (optional)
# ---------------------------------------------------------------------------
def scrape_newsdata(api_key: str, days_back: int) -> pd.DataFrame:
    """Free tier: 200 credits/day. Good for historical coverage."""
    if not api_key:
        return pd.DataFrame()

    log.info("  NewsData.io API...")
    records = []
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsdata.io/api/1/news"
    page = None

    while True:
        params = {
            "apikey": api_key, "q": "bitcoin OR BTC",
            "language": "en", "from_date": cutoff,
            "category": "business,technology",
        }
        if page:
            params["page"] = page
        try:
            data = requests.get(url, params=params, timeout=15).json()
        except Exception as e:
            log.warning(f"  NewsData.io failed: {e}")
            break

        if data.get("status") != "success":
            log.warning(f"  NewsData.io error: {data.get('message', 'unknown')}")
            break

        for article in data.get("results", []):
            pub = article.get("pubDate", "")
            try:
                published = datetime.strptime(pub[:10], "%Y-%m-%d")
            except Exception:
                continue
            title = article.get("title", "") or ""
            desc  = article.get("description", "") or ""
            text  = f"{title}. {desc}"[:512]

            if is_btc_relevant(text):
                records.append({
                    "date": published.date(), "source": "newsdata",
                    "title": title, "summary": desc[:500], "text": text,
                })

        page = data.get("nextPage")
        if not page:
            break
        time.sleep(0.5)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 1F. Collect all news sources
# ---------------------------------------------------------------------------
def collect_all_news() -> pd.DataFrame:
    """
    Gather news from all available sources:
      1. Pre-scraped CSV (fastest, no network needed)
      2. RSS feeds (free, no API key)
      3. CryptoPanic API (optional)
      4. NewsData.io API (optional)

    Deduplicates by title and returns a combined DataFrame.
    """
    cutoff = datetime.now() - timedelta(days=DAYS_BACK)

    frames = [
        load_prescraped_news(),
        scrape_rss(cutoff),
        scrape_cryptopanic(CRYPTOPANIC_TOKEN, DAYS_BACK),
        scrape_newsdata(NEWSDATA_KEY, DAYS_BACK),
    ]
    frames = [f for f in frames if not f.empty]

    if not frames:
        log.error("No news collected from any source.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    log.info(f"Total unique articles: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# 2A. Google Trends (pytrends)
# ---------------------------------------------------------------------------
def fetch_google_trends_daily(days_back: int = TRENDS_DAYS_BACK,
                              keywords: list[str] | None = None,
                              geo: str = TRENDS_GEO) -> pd.DataFrame:
    """
    Fetch daily Google Trends interest for BTC-related keywords.

    Notes:
      - Google Trends often returns daily resolution only for shorter windows.
      - We fetch the last `days_back` days which is sufficient for the model
        evaluation windows + leakage-safe lagging.
    """
    try:
        from pytrends.request import TrendReq
    except Exception as e:
        raise ImportError(
            "pytrends is required for Google Trends. Install with: pip install pytrends"
        ) from e

    if keywords is None:
        keywords = TRENDS_KEYWORDS

    end = datetime.utcnow().date()
    start = end - timedelta(days=days_back)
    timeframe = f"{start.isoformat()} {end.isoformat()}"

    log.info(f"Google Trends: {timeframe} | keywords={keywords}")
    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])

    df = df.reset_index().rename(columns={"date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Standardize column names so they are stable across runs
    rename = {}
    for k in keywords:
        safe = k.lower().strip().replace(" ", "_")
        rename[k] = f"trends_{safe}"
    df = df.rename(columns=rename)

    trend_cols = [c for c in df.columns if c.startswith("trends_")]
    df["trends_mean"] = df[trend_cols].mean(axis=1) if trend_cols else 0.0

    keep = ["date"] + trend_cols + ["trends_mean"]
    df = df[keep].sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2B. Fear & Greed Index (Alternative.me)
# ---------------------------------------------------------------------------
def fetch_fear_greed_daily(limit: int = FGI_LIMIT) -> pd.DataFrame:
    """
    Fetch Crypto Fear & Greed Index time series (daily).

    Source: alternative.me API.
    """
    url = "https://api.alternative.me/fng/"
    params = {"limit": limit, "format": "json"}
    log.info("Fear & Greed Index API...")
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("data", []) or []
    if not rows:
        return pd.DataFrame(columns=["date", "fgi_value", "fgi_value_norm", "fgi_classification"])

    df = pd.DataFrame(rows)
    # timestamp is seconds since epoch
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    df["fgi_value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fgi_value_norm"] = (df["fgi_value"] / 100.0).clip(0, 1)
    df["fgi_classification"] = df.get("value_classification", "").astype(str)

    df = df[["date", "fgi_value", "fgi_value_norm", "fgi_classification"]]
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df
