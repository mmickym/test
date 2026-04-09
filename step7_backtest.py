"""
step7_backtest.py - Quantitative Backtesting (Unified)
=======================================================
Simulates trading on the TEST set (walk-forward window) using model signals.

Strategy logic:
  - If model predicts positive return -> BUY at today's close
  - If model predicts negative return -> SELL (stay in cash)
  - Each trade has a Stop-Loss (-2%) and Take-Profit (+4%) -> risk:reward 1:2
  - Intraday SL/TP checked using High/Low of the NEXT day

Tracks per strategy:
  - Total Return (%)
  - Win Rate (%)
  - Sharpe Ratio (annualized)
  - Max Drawdown (%)
  - Number of trades

Compares: Model A only | Ensemble (A+B) | Buy-and-Hold BTC

Output: Equity curve chart + summary table printed to console.
"""

import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, TRADING_FEE_PCT, BACKTEST_CHART_PATH,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core backtesting engine
# ---------------------------------------------------------------------------
def simulate_strategy(dates, prices_close, prices_high, prices_low,
                      predictions, strategy_name,
                      stop_loss=STOP_LOSS_PCT,
                      take_profit=TAKE_PROFIT_PCT,
                      fee=TRADING_FEE_PCT):
    """
    Simulate a long-only strategy with stop-loss and take-profit.

    Logic for each day t:
      1. Model predicts tomorrow's return at close of day t
      2. If prediction > 0: BUY at close of day t
      3. Next day (t+1): check if High or Low triggers TP or SL
         - If Low  <= entry * (1 + stop_loss):  exit at stop-loss
         - If High >= entry * (1 + take_profit): exit at take-profit
         - Otherwise: exit at close of day t+1
      4. If prediction <= 0: stay in cash (no trade)

    Returns:
        dict with equity curve, trade log, and performance metrics
    """
    n = len(predictions)
    equity = np.ones(n)       # start with $1
    cash = 1.0
    trades = []               # list of trade results

    for t in range(n - 1):
        # Model signal: BUY if predicted return > 0
        if predictions[t] > 0:
            entry_price = prices_close[t]
            next_high   = prices_high[t + 1]
            next_low    = prices_low[t + 1]
            next_close  = prices_close[t + 1]

            sl_price = entry_price * (1 + stop_loss)
            tp_price = entry_price * (1 + take_profit)

            # Check SL/TP using intraday prices
            # Assume worst case: SL checked before TP (conservative)
            if next_low <= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
            elif next_high >= tp_price:
                exit_price = tp_price
                exit_reason = "TP"
            else:
                exit_price = next_close
                exit_reason = "CLOSE"

            # Calculate return after fees
            gross_return = (exit_price / entry_price) - 1
            net_return   = gross_return - 2 * fee  # fee on entry + exit

            cash *= (1 + net_return)
            trades.append({
                "date":        dates[t],
                "entry":       entry_price,
                "exit":        exit_price,
                "return_pct":  net_return * 100,
                "exit_reason": exit_reason,
                "win":         net_return > 0,
            })

        # Record equity at each point
        equity[t + 1] = cash

    # Fill forward the last equity value
    equity[0] = 1.0

    # -- Compute metrics ---------------------------------------------------
    metrics = _compute_metrics(equity, trades, strategy_name)
    metrics["equity"] = equity
    metrics["trades"] = trades
    return metrics


# ---------------------------------------------------------------------------
# Buy-and-hold baseline
# ---------------------------------------------------------------------------
def simulate_buy_and_hold(prices_close):
    """Simple buy-and-hold: invest $1 at start, hold throughout."""
    equity = prices_close / prices_close[0]
    daily_returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] / equity[0] - 1) * 100
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(252)

    running_max = np.maximum.accumulate(equity)
    drawdowns   = (equity - running_max) / running_max
    max_dd      = drawdowns.min() * 100

    return {
        "name":         "Buy & Hold BTC",
        "total_return":  total_return,
        "win_rate":      float("nan"),
        "n_trades":      1,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "equity":        equity,
        "trades":        [],
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def _compute_metrics(equity, trades, name):
    """Calculate performance metrics from equity curve and trade log."""
    total_return = (equity[-1] / equity[0] - 1) * 100
    n_trades     = len(trades)
    wins         = sum(1 for t in trades if t["win"])
    win_rate     = (wins / n_trades * 100) if n_trades > 0 else 0.0

    # Daily returns from equity curve
    daily_ret = np.diff(equity) / (equity[:-1] + 1e-12)
    sharpe    = (np.mean(daily_ret) / (np.std(daily_ret) + 1e-9)) * np.sqrt(252)

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns   = (equity - running_max) / (running_max + 1e-12)
    max_dd      = drawdowns.min() * 100

    return {
        "name":         name,
        "total_return":  total_return,
        "win_rate":      win_rate,
        "n_trades":      n_trades,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
    }


# ---------------------------------------------------------------------------
# Main backtest function (called from main.py)
# ---------------------------------------------------------------------------
def run_backtest(eval_results, model_label=""):
    """
    Run the full backtest using predictions from step6.

    Args:
        eval_results: dict returned by step6_evaluate.evaluate_ensemble()
                      Must contain: df_window, dates, y_true,
                                    pred_a, pred_ensemble, min_train
        model_label:  e.g. "LightGBM" or "XGBoost" (used in chart title/filename)
    """
    suffix = f" ({model_label})" if model_label else ""
    print("\n" + "=" * 60)
    print(f"STEP 7: Quantitative Backtesting{suffix}")
    print("=" * 60)

    df_window    = eval_results["df_window"]
    min_train    = eval_results["min_train"]

    # Use only the walk-forward test period (after min_train warm-up)
    df_test      = df_window.iloc[min_train:].reset_index(drop=True)
    dates        = df_test["Date"].values
    prices_close = df_test["Price"].values
    prices_high  = df_test["High"].values
    prices_low   = df_test["Low"].values
    pred_a       = eval_results["pred_a"][min_train:]
    pred_ens     = eval_results["pred_ensemble"][min_train:]

    print(f"  Test period: {len(df_test)} days "
          f"({pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()})")
    print(f"  Stop-Loss: {STOP_LOSS_PCT*100:.1f}%  |  "
          f"Take-Profit: {TAKE_PROFIT_PCT*100:.1f}%  |  "
          f"Fee: {TRADING_FEE_PCT*100:.2f}%")

    # -- Run all strategies ------------------------------------------------
    results_a   = simulate_strategy(dates, prices_close, prices_high, prices_low,
                                    pred_a, "Model A (Tech Only)")
    results_ens = simulate_strategy(dates, prices_close, prices_high, prices_low,
                                    pred_ens, "Ensemble (A+B)")
    results_bh  = simulate_buy_and_hold(prices_close)

    all_results = [results_a, results_ens, results_bh]

    # -- Print summary table -----------------------------------------------
    _print_summary(all_results)

    # -- Plot equity curves ------------------------------------------------
    chart_path = BACKTEST_CHART_PATH
    if model_label:
        chart_path = chart_path.replace(".png", f"_{model_label.lower()}.png")
    _plot_equity(dates, all_results, model_label, chart_path)

    # -- Print trade breakdown for ensemble --------------------------------
    _print_trade_breakdown(results_ens)

    return all_results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _print_summary(all_results):
    """Print performance comparison table."""
    print("\n  " + "-" * 72)
    print(f"  {'Strategy':25s} {'Return':>9s} {'Win Rate':>10s} "
          f"{'Trades':>8s} {'Sharpe':>8s} {'Max DD':>9s}")
    print("  " + "-" * 72)

    for r in all_results:
        wr = f"{r['win_rate']:.1f}%" if not np.isnan(r.get("win_rate", 0)) else "  n/a"
        print(f"  {r['name']:25s} {r['total_return']:>+8.2f}% {wr:>10s} "
              f"{r['n_trades']:>8d} {r['sharpe']:>+8.2f} {r['max_drawdown']:>+8.2f}%")

    print("  " + "-" * 72)


def _print_trade_breakdown(result):
    """Print SL/TP/CLOSE breakdown for a strategy."""
    trades = result.get("trades", [])
    if not trades:
        return

    sl_count = sum(1 for t in trades if t["exit_reason"] == "SL")
    tp_count = sum(1 for t in trades if t["exit_reason"] == "TP")
    cl_count = sum(1 for t in trades if t["exit_reason"] == "CLOSE")

    print(f"\n  Ensemble trade exits:  "
          f"SL={sl_count}  TP={tp_count}  Close={cl_count}  "
          f"Total={len(trades)}")

    # Average return by exit type
    for reason in ["TP", "SL", "CLOSE"]:
        subset = [t["return_pct"] for t in trades if t["exit_reason"] == reason]
        if subset:
            print(f"    {reason:5s}  avg={np.mean(subset):+.2f}%  "
                  f"count={len(subset)}")


def _plot_equity(dates, all_results, model_label="", chart_path=BACKTEST_CHART_PATH):
    """Plot equity curves for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {"Model A (Tech Only)": "steelblue",
              "Ensemble (A+B)":      "orangered",
              "Buy & Hold BTC":      "gray"}

    for r in all_results:
        c = colors.get(r["name"], "black")
        lw = 2.0 if "Ensemble" in r["name"] else 1.2
        ls = "--" if "Hold" in r["name"] else "-"
        label = f"{r['name']} ({r['total_return']:+.2f}%)"
        ax.plot(dates, r["equity"], label=label, color=c, lw=lw, ls=ls)

    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    title = "Equity Curve Comparison (Walk-Forward Test Period)"
    if model_label:
        title += f" - {model_label}"
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Portfolio Value ($1 start)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n  Equity chart saved -> {chart_path}")
    plt.close(fig)
