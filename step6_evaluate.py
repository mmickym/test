"""
step6_evaluate.py - Ensemble Evaluation & Visualization
=========================================================
Evaluates the two-stage model using:
  1. Walk-forward backtesting (expanding window, re-fit Model B each step)
  2. Empirical quantile-based 95% confidence intervals
  3. Three-panel diagnostic chart

Metrics reported:
  - RMSE, MAE, Directional Accuracy for Model A alone vs Ensemble
  - CI coverage and width
"""

import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from config import (
    NEWS_WINDOW_DAYS, TARGET_COL, EVAL_CHART_PATH,
)

log = logging.getLogger(__name__)

# Suppress convergence warnings from small-window ElasticNet refits
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# Confidence intervals from historical errors
# ---------------------------------------------------------------------------
def compute_confidence_intervals(errors, predictions, confidence=0.95):
    """Non-parametric CIs using empirical quantiles of past errors."""
    alpha = 1 - confidence
    lower_q = np.quantile(errors, alpha / 2)
    upper_q = np.quantile(errors, 1 - alpha / 2)
    return predictions - upper_q, predictions - lower_q


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
def evaluate_ensemble(df, model_a, model_b, scaler_b,
                      tech_features, sent_features, mask,
                      model_label=""):
    """
    Walk-forward backtest on the 65-day news window.

    At each step t:
      - Model A prediction is fixed (trained on pre-window data)
      - Model B is re-fit on an expanding window [0..t-1]
      - The scaler is re-fit each step (no data leakage)
      - Predict day t, record the error

    This simulates how the model would perform if deployed live.
    """
    suffix = f" ({model_label})" if model_label else ""
    print("\n" + "=" * 60)
    print(f"STEP 6: Ensemble Evaluation{suffix}")
    print("=" * 60)

    df_clean  = df[mask].reset_index(drop=True)
    df_window = df_clean.tail(NEWS_WINDOW_DAYS).copy().reset_index(drop=True)

    y_true = df_window[TARGET_COL].values
    dates  = df_window["Date"].values

    # Model A predictions (fixed, truly out-of-sample)
    pred_a = model_a.predict(df_window[tech_features])

    # Walk-forward loop for Model B
    min_train = 15   # minimum rows before Model B starts predicting
    pred_b_wf = np.full(len(df_window), 0.0)

    for t in range(min_train, len(df_window)):
        train_slice = df_window.iloc[:t]
        df_tr = (train_slice[train_slice["has_news"]]
                 if train_slice["has_news"].sum() >= 10
                 else train_slice)

        X_tr = df_tr[sent_features].fillna(0).values
        y_tr = df_tr[TARGET_COL].values - model_a.predict(df_tr[tech_features])

        scaler_wf = StandardScaler()
        X_tr_sc = scaler_wf.fit_transform(X_tr)

        n_splits = min(3, max(2, len(df_tr) // 10))
        model_b_wf = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            alphas=np.logspace(-4, 1, 15),
            cv=TimeSeriesSplit(n_splits=n_splits),
            max_iter=5000, random_state=42,
        )
        model_b_wf.fit(X_tr_sc, y_tr)

        X_test_sc = scaler_wf.transform(
            df_window.iloc[[t]][sent_features].fillna(0).values
        )
        pred_b_wf[t] = model_b_wf.predict(X_test_sc)[0]

    pred_ens_wf = pred_a + pred_b_wf

    # Ensemble with final (non-walk-forward) Model B for comparison
    X_sent_sc = scaler_b.transform(df_window[sent_features].fillna(0))
    pred_ens_final = pred_a + model_b.predict(X_sent_sc)

    # -- Confidence intervals from walk-forward errors ---------------------
    wf_errors = y_true[min_train:] - pred_ens_wf[min_train:]
    ci_lower, ci_upper = compute_confidence_intervals(
        wf_errors, pred_ens_final, confidence=0.95
    )

    # -- Metrics -----------------------------------------------------------
    def report(y, yhat, label, start=0):
        ys, yhs = y[start:], yhat[start:]
        rmse    = np.sqrt(mean_squared_error(ys, yhs))
        mae     = mean_absolute_error(ys, yhs)
        dir_acc = np.mean(np.sign(ys) == np.sign(yhs))
        print(f"  {label:30s}  RMSE={rmse:.6f}  MAE={mae:.6f}  "
              f"Dir.Acc={dir_acc:.2%}")
        return rmse, dir_acc

    print(f"\n  Metrics on walk-forward period (day {min_train}+):")
    rmse_a, da_a  = report(y_true, pred_a,        "Model A (tech only)",    min_train)
    rmse_wf, da_wf = report(y_true, pred_ens_wf,  "Ensemble (walk-fwd)",   min_train)
    _,       _     = report(y_true, pred_ens_final,"Ensemble (final B)",    min_train)

    print(f"\n  RMSE improvement (walk-fwd): {(rmse_a - rmse_wf) / rmse_a * 100:+.1f}%")
    print(f"  Dir.Acc lift (walk-fwd):     {(da_wf - da_a) * 100:+.1f}pp")

    # CI coverage
    in_ci = ((y_true[min_train:] >= ci_lower[min_train:]) &
             (y_true[min_train:] <= ci_upper[min_train:]))
    print(f"\n  95% CI coverage:  {in_ci.mean():.1%}")
    print(f"  Mean CI width:    {np.mean(ci_upper[min_train:] - ci_lower[min_train:]):.6f}")

    # -- Three-panel chart -------------------------------------------------
    chart_path = EVAL_CHART_PATH
    if model_label:
        chart_path = chart_path.replace(".png", f"_{model_label.lower()}.png")
    _plot_results(dates, y_true, pred_a, pred_ens_wf,
                  ci_lower, ci_upper, df_window, model_label, chart_path)

    return {
        "rmse_a": rmse_a, "rmse_ensemble": rmse_wf,
        "dir_acc_a": da_a, "dir_acc_ensemble": da_wf,
        "ci_coverage": float(in_ci.mean()),
        # Pass data downstream for step7 backtesting
        "df_window": df_window,
        "dates": dates,
        "y_true": y_true,
        "pred_a": pred_a,
        "pred_ensemble": pred_ens_wf,
        "min_train": min_train,
    }


def evaluate_model_a_only(df, model_a, features, mask, model_label=""):
    """
    Evaluate a single-stage Model A on the same walk-forward window used elsewhere.

    This mirrors the windowing used by evaluate_ensemble(), but without Model B.
    Returns a dict compatible with step7_backtest.run_backtest() by setting
    pred_ensemble == pred_a.
    """
    suffix = f" ({model_label})" if model_label else ""
    print("\n" + "=" * 60)
    print(f"STEP 6: Model A Only Evaluation{suffix}")
    print("=" * 60)

    df_clean = df[mask].reset_index(drop=True)
    df_window = df_clean.tail(NEWS_WINDOW_DAYS).copy().reset_index(drop=True)

    y_true = df_window[TARGET_COL].values
    dates = df_window["Date"].values

    pred_a = model_a.predict(df_window[features])

    rmse = np.sqrt(mean_squared_error(y_true, pred_a))
    mae = mean_absolute_error(y_true, pred_a)
    dir_acc = np.mean(np.sign(y_true) == np.sign(pred_a))

    print(f"  RMSE={rmse:.6f}  MAE={mae:.6f}  Dir.Acc={dir_acc:.2%}")

    # Use a minimal warm-up so backtest behavior matches evaluate_ensemble downstream
    min_train = 15

    return {
        "rmse_a": rmse,
        "dir_acc_a": dir_acc,
        "df_window": df_window,
        "dates": dates,
        "y_true": y_true,
        "pred_a": pred_a,
        "pred_ensemble": pred_a,
        "min_train": min_train,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_results(dates, y_true, pred_a, pred_ens_wf,
                  ci_lower, ci_upper, df_window,
                  model_label="", chart_path=EVAL_CHART_PATH):
    """Generate the three-panel evaluation chart."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    tag = f" [{model_label}]" if model_label else ""

    # Panel 1: Predictions vs Actual
    ax = axes[0]
    ax.plot(dates, y_true,      label="Actual",                color="black",     lw=1.5)
    ax.plot(dates, pred_a,      label="Model A (tech)",        color="steelblue", lw=1, alpha=0.8)
    ax.plot(dates, pred_ens_wf, label="Ensemble (walk-fwd)",   color="orangered", lw=1.2)
    ax.fill_between(dates, ci_lower, ci_upper,
                    color="orangered", alpha=0.15, label="95% CI")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title(f"Predicted vs Actual Log Returns{tag}")
    ax.legend(fontsize=9)
    ax.set_ylabel("Log Return")

    # Panel 2: Daily Sentiment
    ax = axes[1]
    sent = df_window.get("sentiment_mean", pd.Series([0] * len(dates)))
    colors = ["green" if v > 0 else "red" for v in sent]
    ax.bar(dates, sent, color=colors, alpha=0.6, label="Daily Sentiment")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title("FinBERT Daily Sentiment Mean")
    ax.set_ylabel("Sentiment Score")

    # Panel 3: Residual comparison
    ax = axes[2]
    ax.plot(dates, y_true - pred_a,      label="Model A residuals",    color="steelblue", alpha=0.7)
    ax.plot(dates, y_true - pred_ens_wf, label="Ensemble residuals",   color="orangered",  alpha=0.8)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title("Residual Comparison")
    ax.set_ylabel("Error")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved -> {chart_path}")
    plt.close(fig)
