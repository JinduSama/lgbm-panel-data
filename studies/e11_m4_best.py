"""
E11 - Best formulation on real data: is global LGBM competitive?

Vergleicht die E5-Referenz (Levels + Default-Features) mit den staerksten
eigenen Alternativen und klassischen lokalen Modellen:

    lgbm_levels     : Levels + Default-Features (E5-Referenz, schwaechste Config)
    lgbm_logdiff    : direkte h-Schritt-Log-Aenderung als Label (E6/E2-Gewinner),
                      Ruecktransform Level * exp(pred_change)
    lgbm_ensemble   : gleichgewichtete Mittel beider LGBM-Varianten
    lgbm_perseries  : dieselbe Modellklasse je EINZELSERIE gefittet (kein
                      Cross-Learning -> quantifiziert den Panel-Vorteil)
    autoets / theta : klassische lokale Kontrollen (statsforecast, m=12)
    naive / snaive  : Fixed-Origin-Baselines

Protokoll: Fixed-Origin-Blockprognosen wie im M4-Wettbewerb - jedes Modell
prognostiziert target = fold_end + h vom Origin fold_end (je Serie eigenes
Ende; 2 Folds x 18 Monate). Metriken auf gemeinsamer Unterlage aller Modelle.

Erkenntnis-Ziele:
- Bringt die beste eigene Formulierung das globale LGBM vor die klassischen
  lokalen Modelle?
- Wie gross ist der Cross-Learning-Vorteil (global vs. per-Serie)?
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import metrics_dict, save_fig, save_result
from lgbm_panel.data import load_dataset
from lgbm_panel.experiments import (
    ModelSpec,
    evaluate_predictions,
    expanding_backtest,
    per_series_fold_ends,
)
from lgbm_panel.features import FeatureConfig, TargetTransform, build_supervised
from lgbm_panel.strategies.direct_forecast import DirectLGBM

HORIZONS = (1, 6, 12, 18)
STEP = max(HORIZONS)
N_FOLDS = 2
N_SERIES = 400

LEVELS_CFG = FeatureConfig(exog_cols=[])
LOGDIFF_CFG = FeatureConfig(
    lags=(1, 2, 3, 6, 12, 24),
    rolling_windows=(3, 6, 12),
    rolling_stats=("mean", "std"),
    time_features=("month", "quarter"),
    diff_lags=(),
)

SCHEMA = ["model", "fold", "series", "cutoff", "target_date", "horizon", "y_ref", "y", "pred"]


def _month_ord(s: pd.Series) -> np.ndarray:
    """Monats-Ordinalzahl (jahr*12 + monat) fuer robustes Datums-Matching."""
    return (s.dt.year * 12 + s.dt.month).to_numpy()


def _logdiff_arm(df: pd.DataFrame, fold_end_map: dict[int, pd.Series]) -> pd.DataFrame:
    """Direkte Log-Diff-Prognosen je Fold; Ruecktransform auf Levels."""
    log_df = TargetTransform("log").transform_panel(df)
    sup = build_supervised(log_df, horizons=HORIZONS, config=LOGDIFF_CFG).merge(
        log_df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
        on=["series", "date"],
        how="left",
    )
    lv = df.set_index(["series", "date"])["value"]

    rows = []
    for i in range(1, N_FOLDS + 1):
        fe = fold_end_map[i]
        sup_fe = sup["series"].map(fe)
        train = sup[sup["target_date"] <= sup_fe].copy()
        test = sup[sup["date"] == sup_fe].copy()
        if train.empty or test.empty:
            continue
        # Label: log y[t+h] - log y[cutoff]
        train["y"] = train["y"] - train["y_ref"]
        model = DirectLGBM(horizons=HORIZONS, categorical=("series",))
        model.fit(train, config=LOGDIFF_CFG, num_boost_round=400)
        p = model.predict(test).rename(columns={"date": "cutoff"})
        # Ruecktransform: Level * exp(pred_change) -> Level-Raum.
        anchor = lv.reindex(pd.MultiIndex.from_arrays([p["series"], p["cutoff"]])).to_numpy()
        p["pred"] = anchor * np.exp(np.clip(p["pred"].to_numpy(), -20, 20))
        p["y"] = lv.reindex(
            pd.MultiIndex.from_arrays([p["series"], p["target_date"]])
        ).to_numpy()
        p["y_ref"] = anchor
        p["model"] = "lgbm_logdiff"
        p["fold"] = i
        rows.append(p[SCHEMA])
    return pd.concat(rows, ignore_index=True)


def _perseries_arm(df: pd.DataFrame, fold_end_map: dict[int, pd.Series]) -> pd.DataFrame:
    """Pro Serie ein DirectLGBM (ohne Serien-Feature) - Cross-Learning aus."""
    sup = build_supervised(df, horizons=HORIZONS, config=LEVELS_CFG)
    lv = df.set_index(["series", "date"])["value"]
    frames = []
    n_skipped = 0
    for i in range(1, N_FOLDS + 1):
        fe = fold_end_map[i]
        base = sup.merge(fe.rename("fe"), on="series", how="left")
        train_all = base[base["target_date"] <= base["fe"]]
        test_all = base[base["date"] == base["fe"]]
        for s, tr in train_all.groupby("series"):
            te = test_all[test_all["series"] == s]
            if te.empty:
                continue
            try:
                model = DirectLGBM(horizons=HORIZONS, categorical=())
                model.fit(tr.drop(columns=["fe"]), config=LEVELS_CFG, num_boost_round=150)
                p = model.predict(te.drop(columns=["fe"])).rename(columns={"date": "cutoff"})
            except Exception:
                n_skipped += 1
                continue
            if not np.isfinite(p["pred"].to_numpy()).all():
                continue
            p["y"] = lv.reindex(
                pd.MultiIndex.from_arrays([p["series"], p["target_date"]])
            ).to_numpy()
            p["y_ref"] = lv.reindex(
                pd.MultiIndex.from_arrays([p["series"], p["cutoff"]])
            ).to_numpy()
            p["model"] = "lgbm_perseries"
            p["fold"] = i
            frames.append(p[SCHEMA])
    if n_skipped:
        print(f"per-series arm: {n_skipped} Serien/Folds uebersprungen (zu kurz)")
    return pd.concat(frames, ignore_index=True)


def _statsforecast_arms(
    df: pd.DataFrame, fold_end_map: dict[int, pd.Series], models: dict
) -> pd.DataFrame:
    """Klassische lokale Modelle als Blockprognose vom eigenen fold_end.

    Matching ueber Monats-Ordinals - statsforecast generiert ds im eigenen
    Frequenz-Konvention, nicht zwingend als Monatsende-Stamps.
    """
    from statsforecast import StatsForecast
    for alias, obj in models.items():
        obj.alias = alias

    sf = StatsForecast(models=list(models.values()), freq="MS")
    lv = df.set_index(["series", "date"])["value"]
    frames = []
    for i in range(1, N_FOLDS + 1):
        fe = fold_end_map[i]
        d = df.merge(fe.rename("fe"), on="series", how="left")
        hist = d[d["date"] <= d["fe"]][["series", "date", "value"]].rename(
            columns={"series": "unique_id", "date": "ds", "value": "y"}
        )
        sf.fit(hist)
        fc = sf.predict(h=STEP)
        fc["ord"] = _month_ord(fc["ds"])
        for mname in models:
            f = fc[["unique_id", "ord", mname]].rename(
                columns={"unique_id": "series", "ord": "tord", mname: "pred"}
            )
            # Gewuenschte Ziele: fold_end + h Monate (Monats-Ordinal).
            fe_dates = pd.Series(
                fe.reindex(f["series"].unique()).to_numpy(), index=f["series"].unique()
            )
            fe_ord = fe_dates.dt.year * 12 + fe_dates.dt.month
            parts_h = []
            for h in HORIZONS:
                want = f["series"].map(fe_ord) + h
                sel = f[f["tord"] == want].copy()
                sel["horizon"] = h
                # POSITIONAL: Index der Hilfsserie an sel anpassen, dann
                # kalenderkorrektes DateOffset-Addieren.
                td = fe.reindex(sel["series"])
                td.index = sel.index
                sel["target_date"] = (td + pd.DateOffset(months=h)).to_numpy()
                parts_h.append(sel)
            keep = pd.concat(parts_h, ignore_index=True)
            keep["cutoff"] = keep["series"].map(fe)
            keep["y"] = lv.reindex(
                pd.MultiIndex.from_arrays([keep["series"], keep["target_date"]])
            ).to_numpy()
            keep["y_ref"] = lv.reindex(
                pd.MultiIndex.from_arrays([keep["series"], keep["cutoff"]])
            ).to_numpy()
            keep["model"] = mname
            keep["fold"] = i
            frames.append(keep[SCHEMA])
    return pd.concat(frames, ignore_index=True)


def run() -> dict:
    df = load_dataset("m4", n_series=N_SERIES)
    fold_end_map = per_series_fold_ends(df, n_folds=N_FOLDS, step_months=STEP)

    # Native Arme: Levels-LGBM + Baselines (fixed origin).
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
        parts.append(_perseries_arm(df, fold_end_map))
    except Exception as exc:
        print(f"per-series arm skipped: {exc}")

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

    predictions = pd.concat(parts, ignore_index=True)

    # Ensemble: Mittel aus Levels- und LogDiff-Prognosen je Schluessel.
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

    # --- MASE je Modell ----------------------------------------------------
    def mase_scale(fold_ends: pd.Series) -> pd.Series:
        d = df.merge(fold_ends.rename("fe"), on="series", how="left")
        hist = d[d["date"] <= d["fe"]].sort_values(["series", "date"])
        diff = hist.groupby("series")["value"].diff(12).abs()
        return diff.groupby(hist["series"]).mean()

    scales = {i: mase_scale(fe) for i, fe in fold_end_map.items()}
    rows = []
    for (model, fold, series), grp in predictions.groupby(["model", "fold", "series"]):
        s = scales[int(fold)].get(series, np.nan)
        if not np.isfinite(s) or s <= 0:
            continue
        err = float(np.mean(np.abs(grp["y"] - grp["pred"])))
        rows.append({"model": model, "mase": err / s})
    mase_overall = pd.DataFrame(rows).groupby("model")["mase"].mean().round(4).to_dict()

    # --- Figur --------------------------------------------------------------
    order = [
        "naive",
        "seasonal_naive",
        "autoets",
        "theta",
        "lgbm_perseries",
        "lgbm_levels",
        "lgbm_logdiff",
        "lgbm_ensemble",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for model in order:
        if model not in by_horizon["model"].unique():
            continue
        sub = by_horizon[by_horizon["model"] == model].sort_values("horizon")
        label = {
            "naive": "Naive (fixed)",
            "seasonal_naive": "Seasonal Naive (fixed)",
            "autoets": "AutoETS (lokal)",
            "theta": "Theta (lokal)",
            "lgbm_perseries": "LGBM je Serie",
            "lgbm_levels": "LGBM global, Levels",
            "lgbm_logdiff": "LGBM global, Log-Diff",
            "lgbm_ensemble": "Ensemble (Levels+LogDiff)",
        }.get(model, model)
        style = ":" if model in ("naive", "seasonal_naive") else "-"
        ax.plot(sub["horizon"], sub["mae"], marker="o", ls=style, label=label)
    ax.set_xlabel("Horizont (Monate)")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"E11: M4 fixed-origin Benchmark ({N_SERIES} Serien, {N_FOLDS} Folds)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "e11_m4_best")

    payload = {
        "protocol": {
            "origin": "fixed (Blockprognose vom eigenen fold_end je Serie)",
            "n_folds": float(N_FOLDS),
            "step_months": float(STEP),
            "support": "gemeinsame Nicht-NaN-Unterlage aller Arme",
        },
        "n_series": float(N_SERIES),
        "metrics": metrics_dict(by_horizon),
        "mase_overall": mase_overall,
    }
    save_result("e11_m4_best", payload)
    return payload


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
