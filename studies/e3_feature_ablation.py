"""
E3 - Feature-Ablation: Welche Features beschreiben die Serie?

DGP mit bekanntem exogenem Treiber:
    x_t ~ AR(1)  (der "Treiber", z.B. Marketing-Budget)
    y_t = 100 + 2.2 * x_{t-1} + Saison + leichter Trend + Rauschen

Verglichene Feature-Sets (identischer LGBM, identische Folds):
    lags_only      : nur Target-Lags
    lags_rolling   : + Rolling-Stats
    lags_time      : + Kalenderfeatures
    full_default   : Lags+Rolling+Kalender+Saisondiff (Default-Config)
    full_xs        : + cross-sectionale Panel-Aggregate
    full_exog      : Default + exogener Treiber x (aktuell + Lag-1)

Erkenntnis-Ziele:
1. Wieviel bringt jede Feature-Familie auf diesem DGP?
2. Stimmt die Gain-Importance des vollstaendigen Modells mit der wahren
   DGP-Abhaengigkeit ueberein? (Erster Blick auf Kausalitaet -> E4)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import metrics_dict, save_fig, save_result

from lgbm_panel.experiments import ModelSpec, expanding_backtest
from lgbm_panel.features import FeatureConfig

HORIZONS = (1, 6, 12, 18)


def make_driven_panel(n_series: int = 60, n_periods: int = 132, seed: int = 5) -> pd.DataFrame:
    """Panel, in dem x kausal y antreibt (mit Verzoegerung von einem Monat)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_periods, freq="MS")
    frames = []
    for s in range(n_series):
        t = np.arange(n_periods, dtype=float)
        phase = rng.uniform(0, 2 * np.pi)
        beta = rng.uniform(1.8, 2.6)
        level = rng.uniform(80, 140)

        x = np.zeros(n_periods)
        shocks = rng.normal(0, 6.0, n_periods)
        for i in range(1, n_periods):
            x[i] = 0.7 * x[i - 1] + shocks[i]
        x = np.abs(x)  # Budget-artig: nicht-negativ

        season = 25.0 * np.sin(2 * np.pi * t / 12.0 + phase)
        trend = 0.35 * t
        noise = rng.normal(0, 8.0, n_periods)
        y = level + beta * np.roll(x, 1) + season + trend + noise
        y[0] = level + season[0] + noise[0]

        frames.append(pd.DataFrame({"series": f"S{s:03d}", "date": dates, "value": y, "x": x}))
    return pd.concat(frames, ignore_index=True)


def specs() -> list[ModelSpec]:
    BASE_LAGS = (1, 2, 3, 6, 12, 13, 18)
    return [
        ModelSpec(
            "lags_only",
            config=FeatureConfig(
                lags=BASE_LAGS,
                rolling_windows=(),
                diff_lags=(),
                time_features=(),
                exog_cols=(),
            ),
        ),
        ModelSpec(
            "lags_rolling",
            config=FeatureConfig(
                lags=BASE_LAGS,
                rolling_windows=(3, 6, 12),
                diff_lags=(),
                time_features=(),
                exog_cols=(),
            ),
        ),
        ModelSpec(
            "lags_time",
            config=FeatureConfig(
                lags=BASE_LAGS,
                rolling_windows=(),
                diff_lags=(),
                time_features=("month",),
                exog_cols=(),
            ),
        ),
        ModelSpec(
            "full_default",
            config=FeatureConfig(exog_cols=()),
        ),
        ModelSpec("full_xs", config=FeatureConfig(use_cross_sectional=True, exog_cols=())),
        ModelSpec("full_exog", config=FeatureConfig(exog_cols=("x",))),
    ]


FAMILIES = [
    ("lag_", "Target-Lags"),
    ("roll", "Rolling-Stats"),
    ("diff_", "Saison-Diffs"),
    ("month", "Kalender"),
    ("quarter", "Kalender"),
    ("year", "Kalender"),
    ("xs_", "Cross-Section"),
    ("x", "Treiber x"),
]


def family_of(feature: str) -> str:
    for prefix, name in FAMILIES:
        if feature.startswith(prefix):
            return name
    return feature


def run() -> dict:
    df = make_driven_panel()

    # Metrik-Vergleich aller Feature-Sets.
    res = expanding_backtest(
        df, horizons=HORIZONS, specs=specs(), n_folds=3, step_months=max(HORIZONS)
    )
    m = res.metrics_by_horizon

    # Importance des exogenen Modells (Fold-Mittel, horizontweise).
    res_imp = expanding_backtest(
        df,
        horizons=(12,),
        specs=[specs()[-1]],
        n_folds=3,
        step_months=max(HORIZONS),
        collect_importance=True,
    )
    imp = res_imp.importance.groupby(["feature", "horizon"], as_index=False)["gain"].mean()
    imp["family"] = imp["feature"].map(family_of)
    fam = imp.groupby(["family", "horizon"], as_index=False)["gain"].sum()
    total = fam.groupby("horizon")["gain"].transform("sum")
    fam["share"] = fam["gain"] / total

    # --- Figur -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    piv = m.pivot(index="horizon", columns="model", values="mae")
    order = ["lags_only", "lags_rolling", "lags_time", "full_default", "full_xs", "full_exog"]
    cmap = plt.get_cmap("viridis")
    for j, model in enumerate(order):
        if model not in piv:
            continue
        axes[0].plot(
            piv.index, piv[model], marker="o", color=cmap(j / (len(order) - 1)), label=model
        )
    axes[0].set_title("MAE nach Horizont")
    axes[0].set_xlabel("Horizont (Monate)")
    axes[0].legend(frameon=False, fontsize=8)

    h12 = fam[fam["horizon"] == 12].sort_values("share", ascending=False)
    axes[1].barh(h12["family"], h12["share"], color="#2c7fb8")
    axes[1].set_title("Gain-Anteil je Feature-Familie (h=12)")
    axes[1].set_xlabel("Anteil an Total-Gain")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("E3: Feature-Ablation mit bekanntem Treiber", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "e3_feature_ablation")
    payload = {
        "metrics": metrics_dict(m),
        "importance_share_h12": dict(zip(h12["family"], h12["share"].round(4), strict=True)),
    }
    save_result("e3_feature_ablation", payload)
    return payload


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
