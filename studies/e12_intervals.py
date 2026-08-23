"""
E12 - Prediction intervals: quantile regression vs. split-conformal.

Planungsorientierte Prognosen brauchen Unsicherheitsbaender. Zwei Ansaetze:

    quantile  : ein LightGBM-Booster pro (Horizont, Alpha) mit
                objective=quantile; direkte Alpha-Quantile der Zielgroesse.
    conformal : Split-Conformal um die Punktprognose - SIGNED Residuen-
                Quantile aus einer Kalibrationsmenge (modellfrei).

Beide laufen im LOG-Raum: M4-Serien sind groessenordnungs-heterogen
(Niveaus 1..50000), gepoolte Quantile auf Levels wuerden von den grossen
Serien dominiert und pro Serie massiv miscalibrieren. Die Baender werden
nach der Ruecktransform multiplikativ (exp) und gegen die Level-Wahrheit
ausgewertet.

Setup: 200 zufaellige M4-Monatsserien, fixed-origin Blockprognose wie E11;
innerhalb des Trainings dienen die ersten 75 % der Ziele dem Fit, die
letzten 25 % der Kalibration.

Erkenntnis-Ziele:
- Haelt conformal die nominale Coverage (80 %) ein? (Erwartung: ja,
  leicht konservativ)
- Quantil-Regression: Breite/Pinball im Vergleich.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import save_fig, save_result

from lgbm_panel.data import load_dataset
from lgbm_panel.experiments import per_series_fold_ends
from lgbm_panel.features import FeatureConfig, TargetTransform, build_supervised
from lgbm_panel.strategies.direct_forecast import DirectLGBM

HORIZONS = (1, 6, 12, 18)
STEP = max(HORIZONS)
N_FOLDS = 1
N_SERIES = 200
ALPHAS = (0.1, 0.9)  # 80 %-Intervall
CAL_FRACTION = 0.75

CFG = FeatureConfig(exog_cols=[])


def _signed_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Split-Conformal-Quantil der signierten Residuen (endkorrigiert)."""
    r = residuals[np.isfinite(residuals)]
    n = len(r)
    if n == 0:
        return float("nan")
    level = min(np.ceil((n + 1) * alpha) / n, 1.0)
    return float(np.quantile(r, level))


def _pinball(y_log: np.ndarray, q_log: np.ndarray, alpha: float) -> float:
    d = y_log - q_log
    return float(np.mean(np.where(d >= 0, alpha * d, (alpha - 1.0) * d)))


def run() -> dict:
    df = load_dataset("m4", n_series=N_SERIES)
    fold_end_map = per_series_fold_ends(df, n_folds=N_FOLDS, step_months=STEP)

    # Alles im LOG-Raum (Skalen-Heterogenitaet der M4-Niveaus).
    log_df = TargetTransform("log").transform_panel(df)
    sup = build_supervised(log_df, horizons=HORIZONS, config=CFG)
    lv_level = df.set_index(["series", "date"])["value"]

    fe = fold_end_map[N_FOLDS]
    sup_fe = sup["series"].map(fe)
    train_all = sup[sup["target_date"] <= sup_fe].copy()
    test_all = sup[sup["date"] == sup_fe].copy()
    cal_cut = train_all.groupby("series")["target_date"].transform(
        lambda s: s.sort_values().quantile(CAL_FRACTION)
    )
    train_all["_fit"] = (train_all["target_date"] <= cal_cut).to_numpy()

    rows: list[dict] = []
    for h in HORIZONS:
        tr_h = train_all[train_all["horizon"] == h]
        te_h = test_all[test_all["horizon"] == h]
        if tr_h.empty or te_h.empty:
            continue

        # Punktprognose (Basis fuer conformal)
        pm = DirectLGBM(horizons=(h,), categorical=("series",))
        pm.fit(tr_h[tr_h["_fit"].to_numpy()], config=CFG, num_boost_round=400)
        cal_res = (
            pm.predict(tr_h[~tr_h["_fit"].to_numpy()])
            .rename(columns={"date": "cutoff"})
            .assign(res=lambda d: d["y"] - d["pred"])
        )
        p_point = pm.predict(te_h).rename(columns={"date": "cutoff"})

        key = ["series", "target_date"]
        m = te_h[key + ["y"]].copy()
        m = m.merge(p_point[key + ["pred"]], on=key, how="left")
        # Level-Wahrheit positional (gleiche Zeilenreihenfolge wie m).
        y_true_level = lv_level.reindex(
            pd.MultiIndex.from_arrays([m["series"], m["target_date"]])
        ).to_numpy()
        y_log = m["y"].to_numpy()

        a_lo, a_hi = min(ALPHAS), max(ALPHAS)

        # --- Split-Conformal: signierte Residuen-Quantile -------------------
        q_lo_c = _signed_quantile(cal_res["res"].to_numpy(), a_lo)
        q_hi_c = _signed_quantile(cal_res["res"].to_numpy(), a_hi)
        m["c_lo"], m["c_up"] = m["pred"] + q_lo_c, m["pred"] + q_hi_c

        rows.append({
            "method": "conformal",
            "horizon": h,
            "coverage": float(
                np.mean((y_true_level >= np.exp(m["c_lo"])) & (y_true_level <= np.exp(m["c_up"])))
            ),
            "width": float(np.median(np.exp(m["c_up"]) - np.exp(m["c_lo"]))),
            "pinball": _pinball(y_log, m["c_up"].to_numpy(), a_hi),
        })

        # --- Quantil-Regression ---------------------------------------------
        preds = {}
        for tag, alpha in (("lo", a_lo), ("hi", a_hi)):
            qm = DirectLGBM(
                horizons=(h,),
                params={"objective": "quantile", "alpha": alpha},
                categorical=("series",),
            )
            qm.fit(tr_h[tr_h["_fit"].to_numpy()], config=CFG, num_boost_round=400)
            p = qm.predict(te_h).rename(columns={"date": "cutoff"})
            preds[tag] = p.set_index(key)["pred"]
        m = m.merge(preds["lo"].rename("q_lo"), on=key, how="left")
        m = m.merge(preds["hi"].rename("q_up"), on=key, how="left")

        rows.append({
            "method": "quantile",
            "horizon": h,
            "coverage": float(
                np.mean((y_true_level >= np.exp(m["q_lo"])) & (y_true_level <= np.exp(m["q_up"])))
            ),
            "width": float(np.median(np.exp(m["q_up"]) - np.exp(m["q_lo"]))),
            "pinball": _pinball(y_log, m["q_up"].to_numpy(), a_hi),
        })

    res = pd.DataFrame(rows)

    # --- Figur --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for method, color in (("quantile", "#2c7fb8"), ("conformal", "#d1495b")):
        sub = res[res["method"] == method].sort_values("horizon")
        axes[0].plot(sub["horizon"], sub["coverage"], marker="o", label=method, color=color)
        axes[1].plot(sub["horizon"], sub["width"], marker="o", label=method, color=color)
    axes[0].axhline(0.8, ls="--", color="#555", lw=1)
    axes[0].set_title("Empirische Coverage (Level-Skala, Soll >= 80 %)")
    axes[1].set_title("Median-Intervallbreite (Level-Skala)")
    for ax in axes:
        ax.set_xlabel("Horizont (Monate)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"E12: 80-%-Prognoseintervalle, Log-Raum, M4 ({N_SERIES} Serien)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "e12_intervals")

    payload = {
        "n_series": float(N_SERIES),
        "alphas": list(ALPHAS),
        "cal_fraction": float(CAL_FRACTION),
        "space": "Modelle im Log-Raum; Coverage/Breite auf Level-Skala",
        "rows": res.round(4).to_dict(orient="records"),
    }
    save_result("e12_intervals", payload)
    return payload


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
