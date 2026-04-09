"""
step5_model_b.py - Model B: ElasticNet Sentiment Corrector
============================================================
1. Runs Model A on the 65-day news window to get out-of-sample residuals.
2. Trains ElasticNetCV to predict those residuals using sentiment features.

Why ElasticNet?
  - Only 65 samples -> needs strong regularization
  - L1 (Lasso) component does automatic feature selection
  - L2 (Ridge) component handles correlated sentiment features
  - ElasticNetCV auto-tunes alpha and l1_ratio via TimeSeriesSplit
"""

import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from config import SENTIMENT_FEATURES, TARGET_COL, NEWS_WINDOW_DAYS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 5A. Extract residuals from Model A on the news window
# ---------------------------------------------------------------------------
def extract_residuals(model_a, df, tech_features, mask,
                      news_window=NEWS_WINDOW_DAYS):
    """
    Run Model A on the held-out news window and compute residuals.
    Residuals = actual - predicted (what Model A got wrong).
    These become the target for Model B.
    """
    df_clean = df[mask].copy().reset_index(drop=True)
    df_window = df_clean.tail(news_window).copy()

    X_news   = df_window[tech_features]
    preds_a  = model_a.predict(X_news)
    y_actual = df_window[TARGET_COL].values

    residuals = y_actual - preds_a
    df_window["model_a_pred"] = preds_a
    df_window["residual"]     = residuals

    rmse = np.sqrt(mean_squared_error(y_actual, preds_a))
    print(f"\n  Model A RMSE on news window ({news_window}d): {rmse:.6f}")
    print(f"  Residual mean:  {residuals.mean():.6f}")
    print(f"  Residual std:   {residuals.std():.6f}")
    print(f"  Residual range: [{residuals.min():.4f}, {residuals.max():.4f}]")

    # Stationarity check (optional)
    try:
        from statsmodels.tsa.stattools import adfuller
        adf = adfuller(residuals, autolag="AIC")
        status = "STATIONARY" if adf[1] < 0.05 else "NON-STATIONARY"
        print(f"  ADF test: p={adf[1]:.4f} ({status})")
    except ImportError:
        print("  (Install statsmodels for stationarity test)")

    return df_window


# ---------------------------------------------------------------------------
# 5B. Train Model B
# ---------------------------------------------------------------------------
def train_model_b(df_window, sent_features=None):
    """
    Train ElasticNetCV on sentiment features to predict Model A's residuals.

    Args:
        df_window:     DataFrame from extract_residuals() (has 'residual' column)
        sent_features: List of sentiment column names

    Returns:
        model_b:            Trained ElasticNetCV
        scaler_b:           Fitted StandardScaler
        sent_features_used: List of features actually used
    """
    if sent_features is None:
        sent_features = [f for f in SENTIMENT_FEATURES if f in df_window.columns]

    print("\n" + "=" * 60)
    print("STEP 5: Model B - ElasticNet (Sentiment -> Residuals)")
    print("=" * 60)

    # Only use rows with real news data
    df_with_news = df_window[df_window["has_news"]].copy()
    print(f"  Rows with real news: {len(df_with_news)} / {len(df_window)}")

    if len(df_with_news) < 15:
        print("  WARNING: Very few news rows. "
              "Model B will have limited predictive power.")
        print("  -> Scrape more news data to improve performance.")

    # Fall back to all window rows if not enough news
    df_train = df_with_news if len(df_with_news) >= 15 else df_window

    available_sent = [f for f in sent_features if f in df_train.columns]
    X = df_train[available_sent].fillna(0).values
    y = df_train["residual"].values

    # Scale features (ElasticNet is sensitive to feature scale)
    scaler_b = StandardScaler()
    X_scaled = scaler_b.fit_transform(X)

    # ElasticNetCV with time-aware cross-validation
    n_splits = min(3, max(2, len(df_train) // 15))
    model_b = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        alphas=np.logspace(-4, 1, 15),
        cv=TimeSeriesSplit(n_splits=n_splits),
        max_iter=5000,
        random_state=42,
    )
    model_b.fit(X_scaled, y)

    # Report CV score (primary metric) and in-sample (reference only)
    preds = model_b.predict(X_scaled)
    rmse_in = np.sqrt(mean_squared_error(y, preds))

    best_alpha_idx = np.where(model_b.alphas_ == model_b.alpha_)[0][0]
    best_l1_idx = [0.1, 0.5, 0.9, 1.0].index(model_b.l1_ratio_)
    cv_mse = model_b.mse_path_[best_l1_idx, best_alpha_idx, :].mean()
    rmse_cv = np.sqrt(cv_mse)

    print(f"  Best alpha:    {model_b.alpha_:.6f}")
    print(f"  Best l1_ratio: {model_b.l1_ratio_:.2f}")
    print(f"  RMSE (CV):         {rmse_cv:.6f}")
    print(f"  RMSE (in-sample):  {rmse_in:.6f}  (reference only)")

    # Show which sentiment features have non-zero coefficients
    coef_series = pd.Series(model_b.coef_, index=available_sent)
    nonzero = coef_series[coef_series != 0].sort_values(key=abs, ascending=False)
    print(f"\n  Non-zero sentiment features: {len(nonzero)} / {len(available_sent)}")
    for feat, coef in nonzero.items():
        direction = "(+) bull" if coef > 0 else "(-) bear"
        print(f"    {feat:25s}  coef={coef:+.4f}  {direction}")

    return model_b, scaler_b, available_sent
