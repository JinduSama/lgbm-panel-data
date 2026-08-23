"""
E14 - Foundation models on M4: Chronos-Bolt / Chronos-2 vs. our arms.

Frage: Wo landen schnelle Transformer-Foundation-Modelle im Null-Shot-
Betrieb gegenueber den E11-Armen (LGBM-Formulierungen, AutoETS, Theta,
Naive-Baselines) auf DEMSELBEN Protokoll (fixed-origin Blockprognose je
Serie, gemeinsame Unterlage)?

Arme:
    naive / seasonal_naive  : Fixed-Origin-Baselines
    lgbm_levels             : Levels + Default-Features (trainiert)
    lgbm_logdiff            : direkte Log-Diff-Formulierung (trainiert)
    lgbm_ensemble           : Mittel beider LGBM-Varianten
    autoets / theta         : klassische lokale Modelle (statsforecast)
    chronos-bolt-base       : Null-Shot, univariate, keine Kovariaten
    chronos-2               : Null-Shot, universal ICL (hier univariate)

Kein Training der Foundation-Modelle - nur Inferenz auf der Historie.
"""

import matplotlib.pyplot as plt
import pandas as pd
from _chronos import chronos2_arm, chronos_bolt_arm
from e11_m4_best import (
    HORIZONS,
    N_FOLDS,
    N_SERIES,
    SCHEMA,
    STEP,
    _logdiff_arm,
    _statsforecast_arms,
)

from _common import metrics_dict, save_fig, save_result
from lgbm_panel.data import load_dataset
from lgbm_panel.experiments import (
    ModelSpec,
    evaluate_predictions,
    expanding_backtest,
    per_series_fold_ends,
)

CHRONOS_LABELS = {
    "chronos-bolt-base": "Chronos-Bolt (Null-Shot)",
    "chronos-2": "Chronos-2 (Null-Shot)",
}


def run() -> dict:
    df = load_dataset("m4", n_series=N_SERIES)
    fold_end_map = per_series_fold_ends(df, n_folds=N_FOLDS, step_months=STEP)

    res_native = expanding_backtest(
        df,
        horizons=HORIZONS,
        n_folds=N_FOLDS,
        step_months=STEP,
        origin="fixed",
        specs=[
            ModelSpec(name="lgbm_levels", kind="lgbm"),
            ModelSpec(name="naive", kind="naive"),
            ModelSpec(name="seasonal_naive", kind="snaive"),
        ],
    )
    parts = [res_native.predictions]
    parts.append(_logdiff_arm(df, fold_end_map))

    try:
        from statsforecast.models import AutoETS, Theta

        parts.append(
            _statsforecast_arms(
                df,
                fold_end_map,
                {"autoets": AutoETS(season_length=12), "theta": Theta(season_length=12)},
            )
        )
    except Exception as exc:
        print(f"statsforecast arms skipped: {exc}")


    parts.append(chronos_bolt_arm(df, fold_end_map, HORIZONS, STEP, N_FOLDS, SCHEMA))
    parts.append(chronos2_arm(df, fold_end_map, HORIZONS, STEP, N_FOLDS, SCHEMA))

    predictions = pd.concat(parts, ignore_index=True)

    # Ensemble wie E11: Mittel aus Levels- und LogDiff-Prognosen.
    key_cols = ["fold", "series", "target_date", "horizon"]
    a = predictions[predictions["model"] == "lgbm_levels"].set_index(key_cols)["pred"]
    b = predictions[predictions["model"] == "lgbm_logdiff"].set_index(key_cols)["pred"]
    ens = ((a + b) / 2).rename("pred").reset_index()
    meta = (
        predictions[predictions["model"] == "lgbm_levels"]
        .drop(columns=["pred"])
        .merge(ens, on=key_cols, how="inner")
    )
    meta["model"] = "lgbm_ensemble"
    predictions = pd.concat([predictions, meta[predictions.columns]], ignore_index=True)

    predictions, by_horizon, _ = evaluate_predictions(predictions)

    # --- MASE ---------------------------------------------------------------
    def mase_scale(fold_ends: pd.Series) -> pd.Series:
        d = df.merge(fold_ends.rename("fe"), on="series", how="left")
        hist = d[d["date"] <= d["fe"]].sort_values(["series", "date"])
        diff = hist.groupby("series")["value"].diff(12).abs()
        return diff.groupby(hist["series"]).mean()

    scales = {i: mase_scale(fe) for i, fe in fold_end_map.items()}
    rows = []
    for (model, fold, series), grp in predictions.groupby(["model", "fold", "series"]):
        s = scales[int(fold)].get(series)
        if s is None or not s > 0:
            continue
        err = float((grp["y"] - grp["pred"]).abs().mean())
        rows.append({"model": model, "mase": err / s})
    mase_overall = pd.DataFrame(rows).groupby("model")["mase"].mean().round(4).to_dict()

    # --- Figur --------------------------------------------------------------
    order = [
        "naive",
        "seasonal_naive",
        "autoets",
        "theta",
        "lgbm_levels",
        "lgbm_logdiff",
        "lgbm_ensemble",
        "chronos-bolt-base",
        "chronos-2",
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for model in order:
        if model not in by_horizon["model"].unique():
            continue
        sub = by_horizon[by_horizon["model"] == model].sort_values("horizon")
        label = CHRONOS_LABELS.get(model, {
            "naive": "Naive (fixed)",
            "seasonal_naive": "Seasonal Naive (fixed)",
            "autoets": "AutoETS (lokal)",
            "theta": "Theta (lokal)",
            "lgbm_levels": "LGBM global, Levels",
            "lgbm_logdiff": "LGBM global, Log-Diff",
            "lgbm_ensemble": "Ensemble (Levels+LogDiff)",
        }.get(model, model))
        style = ":" if model in ("naive", "seasonal_naive") else "-"
        lw = 2.4 if model.startswith("chronos") else 1.6
        ax.plot(sub["horizon"], sub["mae"], marker="o", ls=style, lw=lw, label=label)
    ax.set_xlabel("Horizont (Monate)")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"E14: M4 fixed-origin - Foundation Models ({N_SERIES} Serien)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "e14_chronos_m4")

    payload = {
        "protocol": {
            "origin": "fixed (Blockprognose vom eigenen fold_end je Serie)",
            "n_folds": float(N_FOLDS),
            "step_months": float(STEP),
            "support": "gemeinsame Nicht-NaN-Unterlage aller Arme",
            "chronos_mode": "zero-shot inference only, no training/fine-tuning",
        },
        "n_series": float(N_SERIES),
        "metrics": metrics_dict(by_horizon),
        "mase_overall": mase_overall,
    }
    save_result("e14_chronos_m4", payload)
    return payload


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
