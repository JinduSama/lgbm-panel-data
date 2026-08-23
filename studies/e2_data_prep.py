"""
E2 - Datenaufbereitung: Level vs. Log vs. Saison-Differenzen.

Stark trendende Panel-Daten; vier Ziel-Transformationen, identischer LGBM:
    levels        : rohe Werte
    log           : log(y), Ruecktransform via exp
    seasdiff12    : y_t - y_{t-12}, Rekonstruktion mit beobachtetem y[t+h-12]
    log_seasdiff12: log-Differenz (multiplikatives Pendant)

Nur Horizonte <= 12: bei der Saison-Differenz ist die Rekonstruktionsgroesse
y[t+h-12] genau dann noch beobachtet (leakage-frei), wenn h <= 12.

Erkenntnis-Ziel:
- Auf exponentiellen Trends unterschaetzt das Level-Modell systematisch
  (Baum-Modelle extrapolieren nicht ueber den Trainingsbereich).
- Log/Differenzen machen die Serie stationaerer und erlauben saubere
  Extrapolation -> drastisch bessere lange Horizonte.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import save_fig, save_result

from lgbm_panel.data import make_panel
from lgbm_panel.experiments import expanding_backtest

HORIZONS = (1, 6, 12)
COLORS = {
    "levels": "#9d9d9d",
    "log": "#f4a261",
    "seasdiff12": "#2c7fb8",
    "log_seasdiff12": "#7b2cb8",
}


def _panel_lookup(df: pd.DataFrame) -> pd.Series:
    return df.set_index(["series", "date"])["value"]


def _metrics(ok: pd.DataFrame) -> dict[str, float]:
    ok = ok.dropna(subset=["pred", "y"])
    return {
        "mae": float(np.mean(np.abs(ok["y"] - ok["pred"]))),
        "rmse": float(np.sqrt(np.mean((ok["y"] - ok["pred"]) ** 2))),
        "dir_acc": float(
            np.mean(
                np.sign(ok["pred"] - ok["y_ref"]) == np.sign(ok["y"] - ok["y_ref"])
            )
        ),
    }


def run() -> dict:
    raw = make_panel(
        n_series=60,
        n_periods=132,
        horizon=max(HORIZONS),
        seed=21,
        trend_growth=(0.015, 0.035),
        seasonal_strength=(15.0, 35.0),
        noise_scale=(2.0, 5.0),
    )
    lookup = _panel_lookup(raw)

    def backtest(df: pd.DataFrame) -> pd.DataFrame:
        res = expanding_backtest(
            df, horizons=HORIZONS, specs=None, n_folds=3, step_months=max(HORIZONS)
        )
        return res.predictions

    frames: dict[str, pd.DataFrame] = {}

    # --- levels ---------------------------------------------------------
    frames["levels"] = backtest(raw)

    # --- log ------------------------------------------------------------
    log_df = raw.assign(**{"value": np.log(raw["value"].clip(lower=1e-6))})
    p = backtest(log_df)
    for c in ("pred", "y", "y_ref"):
        p[c] = np.exp(p[c])
    frames["log"] = p

    transforms = {
        "seasdiff12": ((lambda v: v), (lambda v: v)),
        "log_seasdiff12": (np.log, np.exp),
    }
    for name, (f, finv) in transforms.items():
        d = raw.copy()
        g = d.groupby("series", sort=False)["value"]
        d["value"] = g.transform(lambda s, f=f: f(s).diff(12))
        d = d.dropna(subset=["value"])
        p = backtest(d)
        key = pd.MultiIndex.from_arrays(
            [p["series"], p["target_date"] - pd.DateOffset(months=12)]
        )
        base = lookup.reindex(key).to_numpy()  # y[t+h-12], beobachtet fuer h<=12
        lvl_origin = lookup.reindex(
            pd.MultiIndex.from_arrays([p["series"], p["cutoff"]])
        ).to_numpy()
        p["pred"] = finv(f(base) + p["pred"])
        p["y"] = finv(f(base) + p["y"])
        p["y_ref"] = lvl_origin
        frames[name] = p

    # --- Metriken je Prep und Horizont ----------------------------------
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name, p in frames.items():
        summary[name] = {}
        for h, grp in p.groupby("horizon"):
            summary[name][str(int(h))] = _metrics(grp)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in frames:
        hs = sorted(summary[name], key=int)
        axes[0].plot(hs, [summary[name][h]["mae"] for h in hs],
                     marker="o", color=COLORS[name], label=name)
        axes[1].plot(hs, [summary[name][h]["dir_acc"] for h in hs],
                     marker="o", color=COLORS[name], label=name)
    axes[0].set_title("MAE nach Horizont")
    axes[1].set_title("Directional Accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlabel("Horizont (Monate)")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("E2: Ziel-Transformation auf stark trendenden Daten", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "e2_data_prep")

    save_result("e2_data_prep", {"scenarios": summary})
    return {"summary": summary}


if __name__ == "__main__":
    import json

    print(json.dumps(run()["summary"], indent=2))
