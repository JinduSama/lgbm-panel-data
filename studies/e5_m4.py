"""
E5 - M4-Monatsdaten: Realer Benchmark.

400 zufaellig gezogene M4-Monatsserien (Wettbewerbshorizont 18 Monate),
Expanding-Window-Backtest mit 2 Folds: globales LGBM vs. Naive-Baselines.

Zusaetzlich zu MAE/sMAPE: MASE (Skalierung mit dem In-Sample-Fehler der
saisonalen Naive, m=12) und Gain-Importance des LGBM.

Erkenntnis-Ziele:
- Uebertragbarkeit der synthetischen Ergebnisse auf echte Daten.
- LGBM-Gewinn ist am kurzen Horizont gross und schrumpft mit h.
- Welche Features tragen auf echten Daten (h=18)?
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import MODEL_COLORS, MODEL_LABELS, MODELS, metrics_dict, save_fig, save_result
from lgbm_panel.data import load_dataset
from lgbm_panel.experiments import expanding_backtest, per_series_fold_ends

HORIZONS = (1, 6, 12, 18)
N_SERIES = 400


def mase_scale(df: pd.DataFrame, fold_ends: pd.Series) -> pd.Series:
    """
    In-Sample-MAE der saisonalen Differenz (m=12) je Serie, nur Historie
    bis zum EIGENEN fold_end der Serie (konsistent zum Backtest-Fenster).
    """
    d = df.merge(fold_ends.rename("fe"), on="series", how="left")
    hist = d[d["date"] <= d["fe"]].sort_values(["series", "date"])
    diff = hist.groupby("series")["value"].diff(12).abs()
    return diff.groupby(hist["series"]).mean()


def run() -> dict:
    df = load_dataset("m4", n_series=N_SERIES)
    res = expanding_backtest(
        df,
        horizons=HORIZONS,
        n_folds=2,
        step_months=max(HORIZONS),
        collect_importance=True,
    )
    m = res.metrics_by_horizon

    # --- MASE aus den Vorhersagen ----------------------------------------
    preds = res.predictions.copy()
    fold_end_map = per_series_fold_ends(df, n_folds=2, step_months=max(HORIZONS))
    scales = {f: mase_scale(df, fe) for f, fe in fold_end_map.items()}
    rows = []
    for (model, fold, series), grp in preds.groupby(["model", "fold", "series"]):
        s = scales[int(fold)].get(series, np.nan)
        if not np.isfinite(s) or s <= 0:
            continue
        err = np.mean(np.abs(grp["y"] - grp["pred"]))
        rows.append({"model": model, "series": series, "mae": float(err), "mase": float(err / s)})
    per_series = pd.DataFrame(rows)
    mase_overall = per_series.groupby("model")["mase"].mean()

    # --- Figur --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    piv = m.pivot(index="horizon", columns="model", values="mae").sort_index()
    for model in MODELS:
        if model in piv:
            axes[0].plot(
                piv.index,
                piv[model],
                marker="o",
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
    axes[0].set_title("MAE nach Horizont")

    smape_piv = m.pivot(index="horizon", columns="model", values="smape").sort_index()
    for model in MODELS:
        if model in smape_piv:
            axes[1].plot(
                smape_piv.index,
                smape_piv[model],
                marker="o",
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
    axes[1].set_title("sMAPE (%) nach Horizont")

    order = mase_overall.sort_values().index
    axes[2].bar(
        [MODEL_LABELS.get(mo, mo) for mo in order],
        [mase_overall[mo] for mo in order],
        color=[MODEL_COLORS.get(mo, "#777777") for mo in order],
    )
    axes[2].axhline(1.0, color="#d1495b", ls="--", lw=1, label="Seasonal-Naive-Niveau")
    axes[2].set_title("MASE (niedriger = besser)")
    axes[2].tick_params(axis="x", labelsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"E5: M4-Monatsdaten ({N_SERIES} Serien, 2 Folds)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "e5_m4_benchmark")

    # --- Importance top10 des LGBM (h=18, Fold-Mittel) --------------------
    imp = None
    if res.importance is not None:
        imp_lgbm = (
            res.importance[res.importance["model"] == "lgbm"]
            .groupby(["feature", "horizon"], as_index=False)["gain"]
            .mean()
        )
        h18 = imp_lgbm[imp_lgbm["horizon"] == 18].copy()
        h18["share"] = h18["gain"] / h18["gain"].sum()
        top = h18.nlargest(10, "share")
        imp = {k: round(float(v), 4) for k, v in zip(top["feature"], top["share"], strict=True)}

    payload = {
        "metrics": metrics_dict(m),
        "mase_overall": {k: round(float(v), 4) for k, v in mase_overall.items()},
        "importance_lgbm_h18_top10": imp,
        "n_series": N_SERIES,
    }
    save_result("e5_m4", payload)
    return payload


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
