"""
presentation_summary.py
======================
Readable console summary for slides / reports: datasets, features, model diagnostics,
and comparison metrics. Run:  python presentation_summary.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import joblib
import pandas as pd

from config import (
    OHLCV_PATH,
    OUTPUT_NEWS_DAILY,
    LEGACY_NEWS_DAILY,
    OUTPUT_MERGED,
    OUTPUTS_DIR,
    MODEL_BUNDLE_PATH,
    FEATURE_LAG_DAYS,
    NEWS_WINDOW_DAYS,
    TECHNICAL_FEATURES,
    SENTIMENT_FEATURES,
    TRENDS_FEATURES,
    FGI_FEATURES,
)

WIDTH = 72


def _hr(ch: str = "-") -> None:
    print(ch * WIDTH)


def _blank(n: int = 1) -> None:
    for _ in range(n):
        print()


def _step_header(step: int, title: str, what: str) -> None:
    _blank(2)
    _hr("=")
    print(f"STEP {step}  {title}")
    _hr("=")
    _blank(1)
    print("Summary (plain English)")
    for line in textwrap.wrap(what, width=WIDTH - 2, initial_indent="  ", subsequent_indent="  "):
        print(line)
    _blank(1)


def _read_csv_any(paths: list[str]) -> pd.DataFrame:
    for p in paths:
        if Path(p).exists():
            return pd.read_csv(p)
    return pd.DataFrame()


def _date_range_from_col(df: pd.DataFrame, col: str) -> tuple[str, str]:
    if df.empty or col not in df.columns:
        return "n/a", "n/a"
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        return "n/a", "n/a"
    return s.min().date().isoformat(), s.max().date().isoformat()


def _feature_presence_in_merged() -> set[str]:
    df = _read_csv_any([OUTPUT_MERGED])
    if df.empty:
        return set()
    return set(df.columns)


def _print_feature_group(
    label: str,
    names: list[str],
    present: set[str] | None,
) -> None:
    print(f"  {label}  ({len(names)} names in config)")
    if present is None:
        blob = ", ".join(names)
    else:
        in_m = [n for n in names if n in present]
        miss = [n for n in names if n not in present]
        print(f"    Present in merged dataset: {len(in_m)} / {len(names)}")
        if miss:
            print(f"    Missing in merged: {', '.join(miss)}")
        blob = ", ".join(in_m)
    if blob.strip():
        wrapped = textwrap.fill(
            blob,
            width=WIDTH - 4,
            initial_indent="    ",
            subsequent_indent="    ",
        )
        print(wrapped)
    _blank(1)


def _print_model_a_importance(bundle: dict) -> None:
    model_a = bundle.get("model_a")
    feats = list(bundle.get("tech_features") or [])
    if model_a is None or not feats:
        print("  (skipped: no Model A in bundle)")
        return
    try:
        imp = getattr(model_a, "feature_importances_", None)
        if imp is None or len(imp) != len(feats):
            print("  (could not read feature importances)")
            return
        s = pd.Series(imp, index=feats).sort_values(ascending=False)
        top_n = 10
        print(f"  Top {min(top_n, len(s))} features by LightGBM importance (from `main.py` bundle):")
        print(f"  {'feature':<22} {'importance':>12}")
        _hr("-")
        for name, val in s.head(top_n).items():
            print(f"  {name:<22} {val:12.4f}")
        if len(s) > top_n:
            rest = s.iloc[top_n:]
            if (rest > 0).any():
                print(f"  ... plus {len(rest)} more (some may be zero)")
            else:
                print(f"  ... remaining {len(rest)} features have 0 importance in this fit")
    except Exception as e:
        print(f"  Error: {e}")


def _print_model_b_coefficients(bundle: dict) -> None:
    model_b = bundle.get("model_b")
    feats = list(bundle.get("sent_features") or [])
    if model_b is None or not feats:
        print("  (skipped: no Model B in bundle)")
        return
    try:
        coef = getattr(model_b, "coef_", None)
        if coef is None or len(coef) != len(feats):
            print("  (could not read coefficients)")
            return
        s = pd.Series(coef, index=feats).sort_values(key=abs, ascending=False)
        print("  ElasticNet on Model A residuals (standardized sentiment inputs):")
        print(f"  {'feature':<22} {'coef':>12}")
        _hr("-")
        for name, val in s.items():
            print(f"  {name:<22} {val:+12.6f}")
        nz = int((s != 0).sum())
        _blank(1)
        print(f"  Non-zero weights: {nz} / {len(s)}  (0 means Lasso dropped that feature)")
    except Exception as e:
        print(f"  Error: {e}")


def _format_metrics_table(df: pd.DataFrame) -> str:
    """Compact string table for key columns."""
    cols = [
        c
        for c in [
            "variant",
            "n_features",
            "rmse",
            "dir_acc",
            "backtest_return_pct",
            "backtest_sharpe",
        ]
        if c in df.columns
    ]
    if not cols:
        return df.to_string(index=False)
    view = df[cols].copy()
    for c in view.columns:
        if c == "variant":
            continue
        if view[c].dtype == float or pd.api.types.is_numeric_dtype(view[c]):
            view[c] = view[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return view.to_string(index=False)


def main() -> None:
    print()
    _hr("=")
    print("BTC/USD + NEWS  |  PRESENTATION SUMMARY")
    _hr("=")
    print()
    print("This script walks through your pipeline in order: data -> features ->")
    print("saved models -> benchmark table. Each STEP ends with a short recap line.")

    present = _feature_presence_in_merged()

    # --- STEP 1 ---
    _step_header(
        1,
        "DATASETS (what you joined)",
        "Price history (OHLCV) is Dataset A. Daily FinBERT sentiment aggregates "
        "are Dataset B. The model learns from rows where both sides align by date.",
    )
    df_ohlcv = _read_csv_any([OHLCV_PATH])
    a_start, a_end = _date_range_from_col(df_ohlcv, "Date")
    print("  A) OHLCV (prices)")
    print(f"      File:     {OHLCV_PATH}")
    print(f"      Dates:    {a_start}  ->  {a_end}")
    print(f"      Rows:     {len(df_ohlcv):,}" if not df_ohlcv.empty else "      Rows:     n/a")

    _blank(1)
    df_sent = _read_csv_any([OUTPUT_NEWS_DAILY, LEGACY_NEWS_DAILY])
    b_start, b_end = _date_range_from_col(df_sent, "date")
    print("  B) Sentiment cache (one row per day after FinBERT in --full mode)")
    print(f"      Preferred: {OUTPUT_NEWS_DAILY}")
    print(f"      Dates:     {b_start}  ->  {b_end}")
    print(f"      Rows:      {len(df_sent):,}" if not df_sent.empty else "      Rows:      n/a")

    _blank(1)
    print("  Recap: You have a long price series and a shorter sentiment series; "
          "missing sentiment days are filled/neutral per pipeline rules.")

    # --- STEP 2 ---
    _step_header(
        2,
        "LEAKAGE CONTROL + EVALUATION WINDOW",
        "External signals (news, trends, fear index) are shifted by one day so "
        "the model does not accidentally use same-day information that would not "
        "be known in advance. The last NEWS_WINDOW_DAYS are used as a strict "
        "out-of-sample style window for some metrics.",
    )
    print(f"  FEATURE_LAG_DAYS     = {FEATURE_LAG_DAYS}  (shift externals backward in time)")
    print(f"  NEWS_WINDOW_DAYS     = {NEWS_WINDOW_DAYS}  (held-out / news-focused window)")
    _blank(1)
    print("  Recap: Conservative timing = harder but more honest numbers.")

    # --- STEP 3 ---
    _step_header(
        3,
        "FEATURE NAMES (what goes into the models)",
        "Technical columns feed LightGBM Model A. Sentiment columns are lagged, "
        "then either added to LightGBM in variant A1 or used in ElasticNet Model B "
        "on residuals. Trends/FGI are optional extra columns if caches exist.",
    )
    if present:
        print(f"  Cross-check vs merged file: {OUTPUT_MERGED}")
    else:
        print(f"  (Merged file not found: {OUTPUT_MERGED} -- run main.py first for column check)")
    _blank(1)

    _print_feature_group("Technical (Model A)", TECHNICAL_FEATURES, present if present else None)
    _print_feature_group("Sentiment (A1 + Model B)", SENTIMENT_FEATURES, present if present else None)
    _print_feature_group("Google Trends (optional)", TRENDS_FEATURES, present if present else None)
    _print_feature_group("Fear & Greed (optional)", FGI_FEATURES, present if present else None)

    print("  Recap: If optional rows show 'Missing in merged', run main.py --full "
          "or fix cache files; A2/A3/A4 will look like A0 until those columns exist.")

    # --- STEP 4 ---
    _step_header(
        4,
        "SAVED MODELS (from python main.py)",
        "The .pkl bundle holds the two-stage model: LightGBM on technicals, then "
        "ElasticNet correcting residuals using sentiment. Importance and coefficients "
        "help you explain which inputs the fit actually used.",
    )
    bundle_path = Path(MODEL_BUNDLE_PATH)
    if not bundle_path.exists():
        print(f"  No bundle at: {MODEL_BUNDLE_PATH}")
        print("  Recap: Run `python main.py` to create it, then re-run this summary.")
    else:
        try:
            bundle = joblib.load(bundle_path)
            _print_model_a_importance(bundle)
            _blank(1)
            _print_model_b_coefficients(bundle)
            _blank(1)
            print("  Recap: Large Model A importance = tree splits used that column often. "
                  "Model B coef near 0 = sentiment did not change the residual much in this fit.")
        except Exception as e:
            print(f"  Could not load bundle: {e}")

    # --- STEP 5 ---
    _step_header(
        5,
        "BENCHMARK TABLE (from compare_models.py)",
        "Rows A0 vs A1 answer: did adding sentiment to the same LightGBM help RMSE "
        "or direction accuracy? TwoStage is a different design (not the same as A1).",
    )
    model_cmp_path = Path(OUTPUTS_DIR) / "model_comparison.csv"
    ablation_path = Path(OUTPUTS_DIR) / "ablation_summary.csv"

    print("  Files:")
    print(f"    - {model_cmp_path}")
    print(f"    - {ablation_path}")
    _blank(1)

    if model_cmp_path.exists():
        df_cmp = pd.read_csv(model_cmp_path)
        if not df_cmp.empty and "variant" in df_cmp.columns:
            want = {"A0_tech_only", "A1_tech_plus_sentiment", "TwoStage"}
            sub = df_cmp[df_cmp["variant"].isin(want)].copy()
            if sub.empty:
                sub = df_cmp
            print("  Key rows (read top to bottom; lower RMSE is better; higher dir_acc is better):")
            _blank(1)
            print(_format_metrics_table(sub))
            _blank(1)
            print("  Column hints:")
            print("    rmse          = error predicting next-day log return (smaller is better)")
            print("    dir_acc       = fraction of correct up/down calls in the test window")
            print("    backtest_*    = simple long/cash backtest (illustrative, not live trading)")
    else:
        print(f"  Missing {model_cmp_path}  ->  run:  python compare_models.py")

    if ablation_path.exists():
        df_ab = pd.read_csv(ablation_path)
        if not df_ab.empty:
            row = df_ab.iloc[0].to_dict()
            _blank(1)
            print("  Sentiment ablation (A1 minus A0), one line:")
            print(f"    Dir.Acc lift (percentage points):  {row.get('dir_acc_lift_pp', 'n/a')}")
            print(f"    RMSE change (%):                   {row.get('rmse_change_pct', 'n/a')}")
            _blank(1)
            print("  Recap: If both are ~0, sentiment did not improve this benchmark on your data.")

    # --- CLOSING ---
    _blank(2)
    _hr("=")
    print("CLOSING CHECKLIST (copy for slides)")
    _hr("=")
    print("  [ ] Say what two datasets are and how dates align + lag rule")
    print("  [ ] Show A0 vs A1 row from model_comparison (main evidence)")
    print("  [ ] Show one chart from outputs/ if you have space")
    print("  [ ] State honestly: prediction != guaranteed profit")
    _blank(1)


if __name__ == "__main__":
    main()
