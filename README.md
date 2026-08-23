# lgbm-panel-data

Guide and utilities for using **LightGBM** for panel / time-series forecasting,
especially monthly series with forecast horizons > 12 months.

## Environment

This project uses [`uv`](https://docs.astral.sh/uv/) as the **sole** Python
environment manager. All dependency and environment management goes through `uv`.

```bash
uv sync            # create venv + install all deps
uv run pytest      # run the test suite
uv run ruff check .   # lint
```

## Studies & Report

Reproducible experiments live in [`studies/`](studies/); results (figures +
JSON metrics) land in `reports/`. The self-contained insight report is
[`reports/report.html`](reports/report.html).

| Script | Question |
|---|---|
| `studies/e1_scenarios.py` | When does global LGBM beat naive baselines? |
| `studies/e2_data_prep.py` | Levels vs. log vs. seasonal differencing on trends |
| `studies/e3_feature_ablation.py` | Which feature families describe the series? |
| `studies/e4_causal.py` | Causal plausibility: intervention study |
| `studies/e5_m4.py` | M4 monthly benchmark (400 series) |
| `studies/e6_levels_vs_logdiff.py` | Levels vs. log-diffs, direct vs. recursive, across five trend regimes |
| `studies/e7_gallery.py` | Forecast gallery: M4 samples + one series per E6 regime |
| `studies/e8_combined.py` | Everything combined: trend + season + driver in one DGP |
| `studies/e9_tuning.py` | Does hyperparameter tuning pay off? (Optuna) |
| `studies/e10_shap_drivers.py` | What drives the forecast? TreeSHAP against the true DGP |
| `studies/e11_m4_best.py` | Best formulation vs. classical local models on M4 (fixed origin) |
| `studies/e12_intervals.py` | Prediction intervals: quantile regression vs. split-conformal |
| `studies/e13_objective_ablation.py` | Training objective ablation (L2/L1/Huber/Quantile) |

Backtest protocol (see the report's methods box): folds are anchored per
series, baselines forecast rolling-origin from each row's own cutoff
(`origin="rolling"`, or `"fixed"` for the classic block protocol), and all
metrics are computed on the common non-NaN support of every model (`n`
column).

```bash
uv run python studies/e1_scenarios.py   # run one study
uv run python studies/build_report.py   # rebuild reports/report.html
```

Core pipeline (`src/lgbm_panel/`): `data` (synthetic + M4 loader with a
parquet cache and length-aware two-digit-year correction), `features` (lags,
rolling, diffs, calendar, exogenous + scenario drivers, cross-sectional
aggregates, `TargetTransform`), `strategies` (global direct LGBM + baselines),
`metrics`, `experiments` (per-series anchored expanding-window backtest with
rolling/fixed origin protocols, common-support evaluation, statsforecast-
compatible prediction schema).
See the bundled skill for the full `uv`-only workflow:

- Project: `.github/skills/uv-only/SKILL.md`
- Personal: `~/.copilot/skills/uv-only/SKILL.md`
