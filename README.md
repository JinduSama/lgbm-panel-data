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

```bash
uv run python studies/e1_scenarios.py   # run one study
uv run python studies/build_report.py   # rebuild reports/report.html
```

Core pipeline (`src/lgbm_panel/`): `data` (synthetic + M4 loader),
`features` (lags, rolling, diffs, calendar, exogenous + scenario drivers,
cross-sectional aggregates), `strategies` (global direct LGBM + baselines),
`metrics`, `experiments` (expanding-window backtest engine).

## Workflow

See the bundled skill for the full `uv`-only workflow:

- Project: `.github/skills/uv-only/SKILL.md`
- Personal: `~/.copilot/skills/uv-only/SKILL.md`
