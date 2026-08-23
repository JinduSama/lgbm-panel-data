# Improvement Plan — lgbm_panel_data

*Review date: 2026-08-23 · Scope: full repo (src/, studies/, reports/, tooling) with focus on the ML methodology.*

## What is already strong (keep)

- **Leakage discipline**: `build_supervised` uses only past-shifted features; the backtest additionally requires `target_date <= fold_end` for training rows. Correct by construction.
- **Direct multi-horizon** formulation with per-horizon boosters; recursion cost honestly quantified (E6: recursive pays ~23× at h=18 on stable exponential trends).
- **Causality thread** is rare and good: E4 (intervention/do-operator) and E10 (TreeSHAP validated against known DGP, additivity checked to 6.5e-13) are genuinely insightful. Finding that SHAP attributes only 3.5 % to the true driver vs 24 % true signal share is a publishable observation.
- **Honest negative results** exist (E9: tuning hurts up to −16 % in one scenario; E8: gains stack).
- Clean package layout, uv-only env, ruff-clean, seeded studies.

---

## P0 — Correctness of conclusions (fix before any new experiment)

### P0.1 Baseline origin-protocol mismatch in `expanding_backtest`

**Problem.** LGBM test rows are *mixed-origin*: a row `(target τ, horizon h)` has cutoff `τ−h`, so its features include data **after** `fold_end` (e.g. h=1 predictions for the last test month use data 17 months past `fold_end`). Baselines are computed **frozen** at `fold_end` (`_baseline_predictions`: `hist = df[df["date"] <= fold_end]`). The comparison is apples-to-oranges and biased toward LGBM at short horizons; baseline labels say "h" but their effective horizon is "months since fold_end".

**Evidence (this repo, trend+seasonal DGP, seed 8, 50 series, 3 folds):**

| h | naive as implemented (frozen) | rolling naive (value at own cutoff) | LGBM | snaive rows kept |
|---|---|---|---|---|
| 1 | 178.97 | **22.05** | 61.12 | 1800 / 2700 |
| 6 | 178.97 | 117.54 | 64.74 | 1800 / 2700 |
| 12 | 178.97 | 213.38 | 62.85 | 1800 / 2700 |
| 18 | 178.97 | 307.30 | 66.91 | 1800 / 2700 |

Two defects visible: (a) frozen naive overstates error 8× at h=1 — with a correct comparator, LGBM *loses* at short horizons on this DGP; (b) snaive drops every row with `target−12 > fold_end` (NaN pred) → only 12 of 18 months evaluated, while LGBM keeps all — different support per model.

Corroboration on real data (E5 JSON): naive MAE is **constant 507.21 across all horizons** — the signature of frozen origin; LGBM MASE 0.893 is reported as "<1 beats seasonal naive", but plain naive achieves **0.836**, i.e. the baseline wins pooled MASE. The report's literature section cites E5 as confirming M5-style "global models dominate"; that claim needs the corrected protocol first.

**Fix (choose one, expose as `origin=` parameter, default `"rolling"`):**
1. Rolling-origin baselines: predict `value[cutoff]` (naive) and `value[target−12]` (snaive) per row, exactly matching each LGBM row's information set; **or**
2. Fixed-origin mode: restrict LGBM test rows to `cutoff == fold_end` so every model forecasts the whole block from one origin (classic M4 protocol).

Plus: evaluate all models on the **intersection** of non-NaN rows per (series, target) or report per-model n alongside metrics.

**Then rerun E1 and E5** and update the report KPI cards + literature cross-check accordingly.

### P0.2 Degenerate directional accuracy for snaive @ h=12

`snaive` predicts `y[t+h−12]`; at h=12 that equals `y_ref`, so `sign(pred−ref)=0` for every row → dir_acc ≈ 0.0 (visible in E5 JSON). Exclude rows with zero predicted direction (like zero true movement already is) or report NA; add a note to `_eval_block`.

### P0.3 `year` as a numeric feature fights extrapolation

Default `FeatureConfig.time_features=("month","quarter","year")`. Trees cannot extrapolate absolute year; it encourages level memorization on trending panels (E5 importance still assigns it 1.3 %). Replace with per-series fractional age **only where a trend formulation needs it**, or drop from defaults; document why.

---

## P1 — ML approach upgrades (highest-value experiments)

Ordered by expected insight-per-effort:

1. **Benchmark the best formulation on real data (new E11).** E5 evaluates the *weakest* configuration (levels + default features). Rerun M4 with direct h-step log-diff (the E2/E6 winner), plus rolling snaive, and classical local controls: ETS / ARIMA / Theta via `statsforecast`. This answers "is global LGBM actually competitive?" honestly — currently unanswerable from the repo.
2. **Local-vs-global control.** The global premise is never tested against per-series fits. Add: per-series LGBM (or per-series ETS as cheap local proxy) and a transfer curve (train global model on k % of series, evaluate on rest). Quantifies the cross-learning benefit the report claims.
3. **Training-objective ablation.** Default is L2 while headline metrics are MAE/sMAPE. Compare `objective ∈ {regression, l1, huber, quantile(α=0.5)}` per scenario. Cheap, likely 0–5 % MAE, and closes a methods gap reviewers would flag.
4. **Prediction intervals (new E12).** Natural continuation for a planning-oriented guide: quantile models at α∈{0.1,0.5,0.9} with pinball loss + empirical coverage per horizon, and/or split-conformal calibration per horizon (model-free wrapper, ~20 lines). None exists today.
5. **Statistical rigor for comparisons.**
   - Multi-seed repeats (≥5 seeds per cell) for E1/E6/E8/E9; today every number is a single-seed point estimate (E9's −16.3 % "tuning hurts" may be noise).
   - More folds: step=max(horizons) with n_folds=2–3 gives tiny eval windows; use overlapping rolling-origin folds (step 6 months, 6–8 folds).
   - Diebold-Mariano (or pooled loss-differential bootstrap) for the headline "X beats Y" claims; report CIs in tables.
6. **Proper early-stopping path.** `valid_fraction` exists but (a) defaults off, (b) splits at a pooled date quantile — wrong when series have unequal lengths (per-series time split needed), (c) never used by any study. Fix the split, wire it into E9 (add `num_boost_round` to the search space; currently fixed 300 rounds interacts with learning-rate tuning).
7. **Target transforms as library citizens.** The log/seas-diff logic is copy-pasted across E2/E6/E7/E8 with hand-rolled inverses — and none applies **retransformation bias correction** (`exp(σ²/2)` smearing or median objective). Extract a `TargetTransform` (identity/log/log1p/seasdiff/Box-Cox, fit/inverse API) into `src/`, add smearing option, and measure its effect at long horizons.
8. **Feature-set upgrades worth an ablation:** Fourier pairs (sin/cos k=1..3) vs raw month (trees split month ordinally); STL-detrended level as feature; recency-weighted samples (`weight ∝ exp(−λ·age)`); interaction constraint between driver x and lags. Skip holidays until a calendar-bearing dataset is added.
9. **Ensemble baseline:** average(levels, direct-logdiff) — near-free accuracy hedge and a nice closing result for the report.
10. **Monotone constraints demo:** for the known-causal driver (E4/E8 worlds), constrain `+beta` monotonicity on x-features; ties the causality theme to a practical LightGBM knob.

---

## P2 — Attribution rigor (extend E10)

- **Explain the decay collapse.** Measured SHAP(x)-slope decays far faster than the φ^(h−1) reference (h=2: 0.67 vs 1.55; ≤0 by h=4) because for h>1 the freshest available x is stale (`x` at cutoff). Add an oracle arm: feed the *true future x path* as scenario features (`exog_scenario_lags`) and show recovery tracks β·φ⁰ — turns a puzzling chart into a mechanism lesson.
- **Correlated-feature attribution:** TreeSHAP is path-dependent here; lags absorb driver credit (budget: Rolling+Lags ≈ 71 %). Cross-check with (a) interventional SHAP (background = marginal/training distribution), (b) `shap_interaction_values` for x×lag pairs, (c) permutation importance on a held-out block.
- **Recovery scatter r=0.46 at h=1:** stratify by season phase/month or use partial dependence of SHAP(x) on x per series to deconfound the season/trend components before fitting slopes.

---

## P3 — Engineering hygiene

| Item | Action |
|---|---|
| **No tests** | README says `uv run pytest`; pyproject sets `testpaths=["tests"]`; directory doesn't exist. Add: leakage regression test (feature timestamps < target), rolling-vs-frozen protocol equivalence test, transform round-trip, MASE/dir_acc edge cases, M4 date-parsing fixtures. This is the highest-leverage engineering task — P0.1 is exactly the class of bug tests would have caught. |
| **Duplicated DGP code** | `make_panel` / `e6._panel` / `e3.make_driven_panel` / `e4.make_world` / `e8.make_world` / `e10.make_world` re-implement the same generators. Consolidate into `lgbm_panel/data/dgp.py` with named scenarios returning `(panel, components)`; studies become thin drivers. |
| **Duplicated backtest harness** | Fold-end math + log-diff backtest loop copy-pasted in E6/E7/E8 (and again in E9). Move into `experiments/` (e.g. `change_forecast_backtest`). |
| **Recursive strategy not in library** | Hand-rolled Python rollout loops live in studies; promote a vectorized `RecursiveLGBM` to `strategies/` and reuse. |
| **M4 loader** | Re-reads the 91 MB CSV on every study; cache a sampled parquet keyed by (n_series, seed); stratify sampling by M4 category; replace the >2016 century heuristic with proper two-digit-year pivot handling. |
| **Dead/stale files** | `studies/_probe_e7.py` (debug leftover), `src/lgbm_panel/plotting/` (unused by studies, writes a fixed `plots/forecast.png`), stale `next_steps.md`, README table missing E8–E10. Delete/update — clean cutover. |
| **Reproducibility metadata** | Results JSONs lack git SHA, timestamps, durations, thread count. Pin LightGBM threads (`num_threads`) for determinism; stamp payloads. |
| **CI** | No workflow. Add `.github/workflows/ci.yml`: ruff check + pytest on push. |
| **Report builder size** | 1195-line HTML-string monolith works; optionally extract section builders if touched anyway — not urgent. |

Minor code notes: `build_supervised`'s `needed = [... if lag <= max_lag]` filter is always-true (dead condition); rolling/diff NaNs are intentionally retained for LightGBM but deserve a comment; `DirectLGBM.fit` pooled-date quantile split covered in P1.6.

---

## Suggested execution order

1. **P0.1–P0.3 + rerun E1/E5 + report update** (protocol correctness gates everything else).
2. **P3 tests + CI** (lock the fixed protocol in place).
3. P1.1–P1.3 (best-formulation M4, local-vs-global, objective ablation) — the substantive ML answers.
4. P1.4–P1.7 (intervals, rigor/multi-seed, early stopping, transforms).
5. P2 attribution extensions; remaining P3 consolidation opportunistically along the way.
