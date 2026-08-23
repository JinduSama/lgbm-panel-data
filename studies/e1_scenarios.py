"""
E1 - Szenario-Raster: Wann schlaegt LGBM die Baselines?

Synthetische Panels mit kontrolliertem DGP (Trend x Saisonalitaet x Rauschen);
Expanding-Window-Backtest von globalem LGBM vs. Naive/Seasonal-Naive.

Erkenntnis-Ziel:
- Struktur (Saisonalitaet) ist der Hebel fuer LGBM-Gewinne.
- Auf strukturllosen, rauschigen Serien gewinnt die simple Baseline.
- Trend + lange Horizonte strafen Level-basierte Modelle (Extrapolation).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from _common import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS,
    metrics_dict,
    metrics_pivot,
    save_fig,
    save_result,
)
from lgbm_panel.data import make_panel
from lgbm_panel.experiments import expanding_backtest

SCENARIOS = {
    "stationaer_saisonal_leise": dict(
        trend_growth=(0.0, 0.0), seasonal_strength=(30.0, 45.0), noise_scale=(1.0, 3.0)
    ),
    "stationaer_saisonal_rauschig": dict(
        trend_growth=(0.0, 0.0), seasonal_strength=(30.0, 45.0), noise_scale=(8.0, 14.0)
    ),
    "trend_saisonal": dict(
        trend_growth=(0.008, 0.02), seasonal_strength=(20.0, 40.0), noise_scale=(2.0, 5.0)
    ),
    "trend_ohne_saison": dict(
        trend_growth=(0.008, 0.02), seasonal_strength=(0.0, 0.0), noise_scale=(2.0, 5.0)
    ),
    "stationaer_strukturlos": dict(
        trend_growth=(0.0, 0.0), seasonal_strength=(0.0, 0.0), noise_scale=(2.0, 4.0)
    ),
}
HORIZONS = (1, 6, 12, 18)


def run() -> dict:
    out: dict[str, dict] = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.ravel()

    for i, (name, kw) in enumerate(SCENARIOS.items()):
        df = make_panel(n_series=50, n_periods=132, horizon=max(HORIZONS), seed=7 + i, **kw)
        res = expanding_backtest(
            df,
            horizons=HORIZONS,
            n_folds=3,
            step_months=max(HORIZONS),
        )
        m = res.metrics_by_horizon
        piv = metrics_pivot(m)

        ax = axes[i]
        for model in MODELS:
            if model in piv.columns:
                ax.plot(
                    piv.index,
                    piv[model],
                    marker="o",
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)

        out[name] = metrics_dict(m)

    # Legende einmal zentral; ungenutzte Subplots aus.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("E1: MAE nach Horizont - Szenario-Raster", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    for ax in axes[len(SCENARIOS) :]:
        ax.axis("off")
    save_fig(fig, "e1_scenario_grid")

    # Relative Verbesserung: LGBM-MAE / SeasonalNaive-MAE (<1 = LGBM besser).
    ratios: dict[str, dict[str, float]] = {}
    for name in SCENARIOS:
        ratios[name] = {
            h: round(
                out[name]["lgbm"][h]["mae"] / max(out[name]["seasonal_naive"][h]["mae"], 1e-9), 3
            )
            for h in map(str, HORIZONS)
        }
    save_result("e1_scenarios", {"metrics": out, "lgbm_over_snaive_mae_ratio": ratios})
    return {"metrics": out, "ratio": ratios}


if __name__ == "__main__":
    print(json.dumps(run()["ratio"], indent=2))
