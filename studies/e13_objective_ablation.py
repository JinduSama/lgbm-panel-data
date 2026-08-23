"""
E13 - Training-objective ablation: L2 vs. L1 vs. Huber vs. Quantile(0.5).

Die Headline-Metriken sind MAE/sMAPE, der Default-Objective aber L2
(Mittelwert-Regression). Konsistent waere Median-Regression - genau das ist
Quantile(alpha=0.5); Huber sitzt dazwischen, L1 ist das harte Median-Ziel.

Setup:
    synthetisch : 3 Szenarien (Rauschen / Trend+Saison / strukturlos), je
                  60 Serien; M4: 150 zufaellige Monatsserien.
Protokoll: expanding_backtest, rolling origin, gemeinsame Unterlage.

Zwei Fallstriicke, die die Studie dokumentiert:
    1. LightGBMs Huber-Delta ("alpha") ist ABSOLUT (Default 0.9). Bei Labels
       >> 1 kollabiert das Modell zu einer quasi-konstanten Prognose. Fair
       nur mit Delta auf Labelskala -> hier 2 x Std der Trainingswerte.
    2. Quantile(0.5) ist in LightGBM mathematisch L1 (identische Gradienten
       bis auf Skalierung -> identische Baeume); als Bestaetigung gefuehrt.

Erkenntnis-Ziel: Kauft man mit einem robusten Objective messbar MAE?
(Literatur-Erwartung: klein, 0-5 %, am ehesten bei Rauschen/Ausreissern.)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from _common import save_fig, save_result

from lgbm_panel.data import make_panel
from lgbm_panel.data.load import load_dataset
from lgbm_panel.experiments import ModelSpec, evaluate_predictions, expanding_backtest

HORIZONS = (1, 6, 12, 18)

SYN_SCENARIOS = {
    "stationaer_saisonal_rauschig": dict(
        trend_growth=(0.0, 0.0), seasonal_strength=(30.0, 45.0), noise_scale=(8.0, 14.0)
    ),
    "trend_saisonal": dict(
        trend_growth=(0.008, 0.02), seasonal_strength=(20.0, 40.0), noise_scale=(2.0, 5.0)
    ),
    "stationaer_strukturlos": dict(
        trend_growth=(0.0, 0.0), seasonal_strength=(0.0, 0.0), noise_scale=(2.0, 4.0)
    ),
}

ROBUST_COLORS = {"lgbm_l1": "#d1495b", "lgbm_huber": "#f4a261", "lgbm_quantile50": "#2c7fb8"}


def _objective_specs(df: pd.DataFrame) -> list[ModelSpec]:
    """Objective-Arme; Huber-Delta skaliert auf die Labelverteilung."""
    huber_alpha = float(2.0 * df["value"].std())
    return [
        ModelSpec(name="lgbm_l2", kind="lgbm", num_boost_round=300),
        ModelSpec(name="lgbm_l1", kind="lgbm", num_boost_round=300, params={"objective": "l1"}),
        ModelSpec(
            name="lgbm_huber",
            kind="lgbm",
            num_boost_round=300,
            params={"objective": "huber", "alpha": round(huber_alpha, 3)},
        ),
        ModelSpec(
            name="lgbm_quantile50",
            kind="lgbm",
            num_boost_round=300,
            params={"objective": "quantile", "alpha": 0.5},
        ),
    ]


def _run_panel(df: pd.DataFrame) -> pd.DataFrame:
    res = expanding_backtest(
        df,
        horizons=HORIZONS,
        n_folds=2,
        step_months=max(HORIZONS),
        specs=_objective_specs(df),
    )
    _, by_horizon = evaluate_predictions(res.predictions)[:2]
    return by_horizon


def _ratio_table(piv: pd.DataFrame) -> dict[str, dict[str, float]]:
    base = piv["lgbm_l2"]
    return {
        model: {
            str(int(h)): round(float(piv.loc[h, model] / max(base.loc[h], 1e-9)), 4)
            for h in piv.index
        }
        for model in piv.columns
    }


def run() -> dict:
    out: dict[str, dict] = {}

    # --- Synthetische Szenarien -------------------------------------------
    for name, kw in SYN_SCENARIOS.items():
        df = make_panel(n_series=60, n_periods=132, horizon=max(HORIZONS), seed=7, **kw)
        m = _run_panel(df)
        piv = m.pivot(index="horizon", columns="model", values="mae")
        out[name] = {
            "mae": {
                model: {str(int(h)): round(float(piv.loc[h, model]), 3) for h in piv.index}
                for model in piv.columns
            },
            "_ratio_vs_l2": _ratio_table(piv),
        }

    # --- M4 -----------------------------------------------------------------
    df_m4 = load_dataset("m4", n_series=150)
    m = _run_panel(df_m4)
    piv = m.pivot(index="horizon", columns="model", values="mae")
    out["m4"] = {
        "mae": {
            model: {str(int(h)): round(float(piv.loc[h, model]), 2) for h in piv.index}
            for model in piv.columns
        },
        "_ratio_vs_l2": _ratio_table(piv),
    }

    # --- Figur: Ratio-Kurven -------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (name, data) in zip(axes, out.items(), strict=True):
        ratios = data["_ratio_vs_l2"]
        for model, color in ROBUST_COLORS.items():
            if model in ratios:
                hs = sorted(ratios[model], key=int)
                ax.plot(
                    [int(h) for h in hs],
                    [ratios[model][h] for h in hs],
                    marker="o",
                    color=color,
                    label=model.replace("lgbm_", ""),
                )
        ax.axhline(1.0, color="#999", ls="--", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Horizont")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAE-Ratio vs. L2 (<1 = besser)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("E13: Trainings-Objective vs. MAE", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "e13_objective_ablation")

    payload = {"scenarios": out}
    save_result("e13_objective_ablation", payload)
    return payload


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
