"""
E15 - Budget plans as future-known covariates: can foundation models use them?

Das Kernszenario der Planungspraxis: ein exogener Treiber (z.B. Budgetplan)
ist fuer den Prognosehorizont BEKANNT. Unsere Studien E4/E8 zeigen, dass ein
LGBM mit Szenario-Features den Treiber voll ausnutzt (Directional Accuracy
49 % -> 93 % unter Intervention). Foundation-Modelle: Chronos-Bolt ist
univariate und kann Kovariaten prinzipiell nicht nutzen; Chronos-2 nimmt
bekannte Zukunftskovariaten nativ via ``future_df`` entgegen.

DGP (60 Serien x 132 Monate):
    x[t] = 0.9 * x[t-1] + (1-0.9)*45 + eps_x        (persistenter Plan-Treiber)
    y[t] = level*exp(g*t) + beta*x[t-1] + amp*sin(2*pi*t/12+phase) + eps_y

Arme (fixed-origin Blockprognose je Serie, wie E11/E14):
    naive / seasonal_naive        : Referenzen
    lgbm_levels                   : ohne Treiber (Ablation)
    lgbm_levels+x_plan            : mit x am Zieldatum (Szenario-Feature)
    chronos-bolt-base             : univariate - ignoriert x zwangslaeufig
    chronos-2                     : univariate gefuettert
    chronos-2+x_plan              : x als bekannte Zukunftskovariate

Erkenntnis-Ziele:
- Wie gross ist der Vorteil des geplanten Treibers im LGBM? (E8-Muster)
- Kann Chronos-2 ihn Null-Shot ueber die Kovariaten-Schnittstelle
  einfangen? Wie nah kommt es dem trainierten LGBM?
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _chronos import chronos2_arm, chronos_bolt_arm
from e11_m4_best import HORIZONS, N_FOLDS, SCHEMA, STEP

from _common import metrics_dict, save_fig, save_result
from lgbm_panel.experiments import ModelSpec, evaluate_predictions, per_series_fold_ends
from lgbm_panel.features import FeatureConfig
from lgbm_panel.strategies.direct_forecast import DirectLGBM

N_SERIES = 60
N_PERIODS = 132


def make_world(seed: int = 42) -> pd.DataFrame:
    """Trend + Saison + persistenter Plan-Treiber x (wirkt auf y[t])."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=N_PERIODS, freq="MS")
    frames = []
    for s in range(N_SERIES):
        level = rng.uniform(80, 400)
        growth = rng.uniform(0.04, 0.11) / 12.0  # pro Monat
        amp = rng.uniform(10.0, 25.0)
        phase = rng.uniform(0, 2 * np.pi)
        beta = rng.uniform(2.5, 6.0)

        t = np.arange(N_PERIODS, dtype=float)
        x = np.empty(N_PERIODS)
        x[0] = rng.uniform(35, 55)
        shocks = rng.normal(0, 3.0, N_PERIODS)
        for i in range(1, N_PERIODS):
            x[i] = 0.9 * x[i - 1] + 0.1 * 45.0 + shocks[i]
        season = amp * np.sin(2 * np.pi * t / 12.0 + phase)
        y = level * np.exp(growth * t) + beta * np.roll(x, 1) + season + rng.normal(
            0, rng.uniform(1.5, 4.0), N_PERIODS
        )
        frames.append(pd.DataFrame({"series": f"S{s:03d}", "date": dates, "value": y, "x": x}))
    return pd.concat(frames, ignore_index=True)


def _lgbm_arm(df: pd.DataFrame, fold_end_map, with_driver: bool) -> pd.DataFrame:
    """Levels-LGBM; mit Treiber als Szenario-Feature (x am Zieldatum)."""
    from lgbm_panel.features import build_supervised

    cfg = FeatureConfig(
        exog_cols=("x",) if with_driver else (),
        exog_lags=(),
        exog_scenario_lags=(0,) if with_driver else (),
    )
    sup = build_supervised(df, horizons=HORIZONS, config=cfg)
    lv_level = df.set_index(["series", "date"])["value"]
    frames = []
    for i in range(1, N_FOLDS + 1):
        fe = fold_end_map[i]
        sup_fe = sup["series"].map(fe)
        train = sup[sup["target_date"] <= sup_fe]
        test = sup[sup["date"] == sup_fe]
        model = DirectLGBM(horizons=HORIZONS, categorical=("series",))
        model.fit(train, config=cfg, num_boost_round=400)
        p = model.predict(test).rename(columns={"date": "cutoff"})
        # Levels-Formulierung: keine Ruecktransform.
        p["y"] = lv_level.reindex(
            pd.MultiIndex.from_arrays([p["series"], p["target_date"]])
        ).to_numpy()
        p["y_ref"] = lv_level.reindex(
            pd.MultiIndex.from_arrays([p["series"], p["cutoff"]])
        ).to_numpy()
        p["model"] = "lgbm_levels_x" if with_driver else "lgbm_levels"
        p["fold"] = i
        frames.append(p[SCHEMA])
    return pd.concat(frames, ignore_index=True)


def run() -> dict:
    df = make_world()
    fold_end_map = per_series_fold_ends(df, n_folds=N_FOLDS, step_months=STEP)

    res_native = expanding_backtest_fixed(df, fold_end_map)

    parts = [res_native]
    parts.append(_lgbm_arm(df, fold_end_map, with_driver=True))
    parts.append(_lgbm_arm(df, fold_end_map, with_driver=False))

    parts.append(chronos_bolt_arm(df, fold_end_map, HORIZONS, STEP, N_FOLDS, SCHEMA))
    parts.append(chronos2_arm(df, fold_end_map, HORIZONS, STEP, N_FOLDS, SCHEMA))
    parts.append(
        chronos2_arm(
            df, fold_end_map, HORIZONS, STEP, N_FOLDS, SCHEMA,
            exog_col="x", name="chronos-2-x",
        )
    )

    predictions = pd.concat(parts, ignore_index=True)
    predictions, by_horizon, _ = evaluate_predictions(predictions)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    styles = {
        "naive": (":", "#9d9d9d"),
        "seasonal_naive": (":", "#f4a261"),
        "lgbm_levels": ("--", "#7b7b7b"),
        "lgbm_levels_x": ("-", "#2c7fb8"),
        "chronos-bolt-base": ("-", "#d1495b"),
        "chronos-2": ("-", "#e07a5f"),
        "chronos-2-x": ("-", "#3a7d44"),
    }
    labels = {
        "naive": "Naive (fixed)",
        "seasonal_naive": "Seasonal Naive (fixed)",
        "lgbm_levels": "LGBM ohne Treiber",
        "lgbm_levels_x": "LGBM + Budgetplan (Szenario)",
        "chronos-bolt-base": "Chronos-Bolt (univariat)",
        "chronos-2": "Chronos-2 (univariat)",
        "chronos-2-x": "Chronos-2 + Budgetplan (Kovariate)",
    }
    for model in styles:
        if model not in by_horizon["model"].unique():
            continue
        sub = by_horizon[by_horizon["model"] == model].sort_values("horizon")
        ls, color = styles[model]
        lw = 2.4 if model in ("lgbm_levels_x", "chronos-2-x") else 1.6
        ax.plot(sub["horizon"], sub["mae"], marker="o", ls=ls, color=color, lw=lw,
                label=labels[model])
    ax.set_xlabel("Horizont (Monate)")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"E15: Budgetplan als Zukunftskovariate ({N_SERIES} Serien)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "e15_chronos_exog")

    payload = {
        "protocol": {
            "origin": "fixed (Blockprognose vom eigenen fold_end je Serie)",
            "n_folds": float(N_FOLDS),
            "step_months": float(STEP),
            "driver": "x wirkt auf y[t], Pfad zum Zielzeitpunkt als bekannt angenommen (Plan)",
        },
        "n_series": float(N_SERIES),
        "metrics": metrics_dict(by_horizon),
    }
    save_result("e15_chronos_exog", payload)
    return payload


def expanding_backtest_fixed(df: pd.DataFrame, fold_end_map) -> pd.DataFrame:
    """Naive/SNaive fixed-origin Arme; Rueckgabe direkt die Vorhersagen."""
    from lgbm_panel.experiments import expanding_backtest

    res = expanding_backtest(
        df,
        horizons=HORIZONS,
        n_folds=N_FOLDS,
        step_months=STEP,
        origin="fixed",
        specs=[
            ModelSpec(name="naive", kind="naive"),
            ModelSpec(name="seasonal_naive", kind="snaive"),
        ],
    )
    return res.predictions


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
