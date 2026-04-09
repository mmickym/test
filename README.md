## BTC/USD + News Sentiment (Implementation Project)

End-to-end Python pipeline: **daily OHLCV** joined with **FinBERT news sentiment** (plus optional externals), time-aligned with **1-day lag** to limit leakage.

### Focused research question
**Does lagged news sentiment improve next-day log-return prediction vs a technical-only baseline?**  
Primary evidence: **A0 (tech-only) vs A1 (tech + sentiment)** in `compare_models.py`.

### Required outputs (what to show your professor)
| Output | What it answers |
|--------|-----------------|
| `outputs/ablation_summary.csv` | Sentiment **lift** (A1 vs A0): Dir.Acc / RMSE |
| `outputs/model_comparison.csv` | One table: all variants; sort by `dir_acc` desc, then `rmse` asc |
| `outputs/btc_two_stage_*.png` / `btc_backtest_*.png` (from `main.py`) | Optional visuals for the write-up |

### Minimal run (after `pip install -r requirements.txt`)
1. First-time data + FinBERT cache: `python main.py --full`  
2. Train two-stage pipeline + save bundle: `python main.py`  
3. **Main comparison (A0 vs A1 + others):** `python compare_models.py`

### Data you must have
- `Bitcoin Historical Data.csv` (OHLCV)
- `news_prescraped_seed.csv` or live scrape → cached sentiment under `cache/`

### Optional (extra experiments — not required for the core question)
- `python direction_model_comparison.py` → `outputs/direction_model_comparison.csv`
- `python presentation_summary.py` → console summary: feature names, Model A importance, Model B coefficients, A0/A1/TwoStage metrics
- `python predict_daily.py` → one-day-ahead forecast (needs `outputs/btc_two_stage_bundle.pkl`)

### Full refresh
`python main.py --full` when you need to re-scrape and re-score news.

More detail: `report.md`.
