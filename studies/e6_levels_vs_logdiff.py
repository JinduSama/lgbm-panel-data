"""
E6 - Level- vs. Log-Differenz-Prognosen: direkte Formulierung vs. Rekursion.

Fragestellung: Sind direkte Level-Forecasts besser als Log-Differenz-
Forecasts? Und was kostet die Rekursivitaet bei Differenzen?

Vier Varianten auf identischem DGP (starker exponentieller Trend + Saison):
    direct_level    : ein LGBM pro Horizont auf rohen Levels
    seasonal_naive  : Referenz-Baseline
    direct_logdiff  : ein LGBM pro Horizont auf der h-Schritt-Log-Aenderung
                      log y[t+h] - log y[t]  -> KEINE Rekursion noetig,
                      Rekonstruktion exp(log y[t] + pred)
    recursive_logdiff: EIN Modell fuer 1-Schritt-Log-Aenderungen, das fuer
                      h Schritte mit den eigenen Vorhersagen weiterspielt
                      (Lags/Rolling werden aus Prognosen fortgeschrieben)

Erkenntnis-Ziele:
- Levels: Extrapolationsproblem bei Trend (siehe E2).
- Direkte Log-Differenzen: stabil ueber alle Horizonte, kein Schneeball.
- Rekursive Log-Differenzen: Fehler akkumulieren mit dem Horizont, weil jede
  Stufe mit der Unsicherheit der vorherigen Prognosen rechnet.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lgbm_panel.data import make_panel
from lgbm_panel.experiments import ModelSpec, expanding_backtest
from lgbm_panel.features import FeatureConfig, build_supervised
from lgbm_panel.strategies import DirectLGBM

from _common import save_fig, save_result

HZ = (1, 3, 6, 12, 18)
N_FOLDS = 3

# Feature-Set des rekursiven Modells (muss im Rollout manuell gespiegelt werden).
R_LAGS = (1, 2, 3, 6, 12)
R_WINDOWS = (3, 12)
RCFG = FeatureConfig(
    lags=R_LAGS,
    rolling_windows=R_WINDOWS,
    rolling_stats=("mean",),
    diff_lags=(),
    time_features=("month",),
    exog_cols=(),
)


def _fold_ends(grid: pd.DatetimeIndex, step: int) -> list[pd.Timestamp]:
    """Identische Fold-Grenzen wie expanding_backtest."""
    return [
        grid[-1] - pd.DateOffset(months=(N_FOLDS - k + 1) * step)
        for k in range(1, N_FOLDS + 1)
    ]


def _row_from_history(hist: np.ndarray, date: pd.Timestamp) -> dict[str, float]:
    """Feature-Zeile im build_supervised-Semantik von RCFG."""
    row: dict[str, float] = {}
    for k in R_LAGS:
        row[f"lag_{k}"] = hist[-k]
    for w in R_WINDOWS:
        row[f"roll{w}_mean"] = float(np.mean(hist[-w:]))
    row["month"] = date.month
    return row


def run() -> dict:
    raw = make_panel(
        n_series=60,
        n_periods=144,
        horizon=max(HZ),
        seed=33,
        trend_growth=(0.012, 0.03),
        seasonal_strength=(15.0, 35.0),
        noise_scale=(2.0, 5.0),
    )
    log_df = raw.assign(value=np.log(raw["value"].clip(lower=1e-6)))
    grid = pd.DatetimeIndex(sorted(raw["date"].unique()))
    ends = _fold_ends(grid, max(HZ))
    level_lookup = raw.set_index(["series", "date"])["value"]

    # ---------------------------------------------------------- 1) Engine-Modelle
    engine = expanding_backtest(
        raw,
        horizons=HZ,
        specs=[ModelSpec("direct_level"), ModelSpec("seasonal_naive", kind="snaive")],
        n_folds=N_FOLDS,
        step_months=max(HZ),
    )

    # ---------------------------------------------------- 2) Direct auf Log-Diff
    sup = build_supervised(log_df, horizons=HZ).merge(
        log_df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
        on=["series", "date"],
        how="left",
    )
    # y und pred sind nach Training Log-Aenderungen; Rekonstruktion unten.
    sup["y_change"] = sup["y"] - sup["y_ref"]
    direct_rows = []
    for i, fe in enumerate(ends, start=1):
        hi = fe + pd.DateOffset(months=max(HZ))
        train = sup[sup["target_date"] <= fe].dropna(subset=["y_change"])
        test = sup[(sup["target_date"] > fe) & (sup["target_date"] <= hi)]
        if train.empty or test.empty:
            continue
        m = DirectLGBM(horizons=HZ).fit(
            train.assign(y=train["y_change"]), num_boost_round=300
        )
        p = m.predict(test)
        p["level_pred"] = np.exp(p["y_ref"] + p["pred"])
        p["truth"] = np.exp(p["y"] + p["y_ref"])
        p["fold"] = i
        direct_rows.append(p)
    direct_ld = pd.concat(direct_rows, ignore_index=True)

    # ------------------------------------------------- 3) Recursive auf Log-Diff
    sup1 = build_supervised(log_df, horizons=(1,), config=RCFG).merge(
        log_df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
        on=["series", "date"],
        how="left",
    )
    sup1 = sup1.dropna(subset=["y_ref"])
    rec_model = DirectLGBM(horizons=(1,)).fit(
        sup1.assign(y=sup1["y"] - sup1["y_ref"]), num_boost_round=300
    )
    feat_cols = list(_row_from_history(np.zeros(24), pd.Timestamp("2020-01-01")))

    recursive_rows = []
    log_hist = {
        s: g.sort_values("date")["value"].to_numpy()
        for s, g in log_df.groupby("series", sort=False)
    }
    for i, fe in enumerate(ends, start=1):
        targets = grid[(grid > fe) & (grid <= fe + pd.DateOffset(months=max(HZ)))]
        for s, hist_full in log_hist.items():
            n_obs = int((grid <= fe).sum())
            base_hist = list(map(float, hist_full[:n_obs]))
            v_end = base_hist[-1]
            hist = base_hist.copy()
            cur = fe
            changes: list[float] = []
            for h in range(1, len(targets) + 1):
                row = _row_from_history(np.asarray(hist), cur)
                x = pd.DataFrame([row])[feat_cols]
                d = float(rec_model.models[1].predict(x)[0])
                changes.append(d)
                hist.append(v_end + sum(changes))
                cur = cur + pd.DateOffset(months=1)
                tgt = targets[h - 1]
                recursive_rows.append({
                    "series": s,
                    "target_date": tgt,
                    "horizon": h,
                    "level_pred": float(np.exp(v_end + sum(changes[:h]))),
                    "truth": float(level_lookup[(s, tgt)]),
                    "y_ref": float(np.exp(v_end)),
                    "fold": i,
                })
    recursive_ld = pd.DataFrame(recursive_rows)

    # ------------------------------------------------------------------ Auswertung
    def evaluate(preds: pd.DataFrame, name: str) -> dict:
        out = {}
        for h, grp in preds.groupby("horizon"):
            ok = grp.dropna(subset=["level_pred", "truth"])
            out[str(int(h))] = {
                "mae": float(np.mean(np.abs(ok["truth"] - ok["level_pred"]))),
                "rmse": float(np.sqrt(np.mean((ok["truth"] - ok["level_pred"]) ** 2))),
                "dir_acc": float(
                    np.mean(
                        np.sign(ok["level_pred"] - ok["y_ref"])
                        == np.sign(ok["truth"] - ok["y_ref"])
                    )
                ),
            }
        return {name: out}

    summary: dict[str, dict] = {}
    summary.update(
        {
            m: {
                str(int(h)): {k: v for k, v in row.items()}
                for h, row in grp.set_index("horizon")
                [["mae", "rmse", "smape", "dir_acc"]].to_dict("index").items()
            }
            for m, grp in engine.metrics_by_horizon.groupby("model")
        }
    )
    summary.pop("naive", None)
    summary.update(evaluate(direct_ld, "direct_logdiff"))
    summary.update(evaluate(recursive_ld, "recursive_logdiff"))

    # ------------------------------------------------------------------ Figur
    colors = {
        "direct_level": "#9d9d9d",
        "seasonal_naive": "#f4a261",
        "recursive_logdiff": "#d1495b",
        "direct_logdiff": "#00798c",
    }
    labels = {
        "direct_level": "Direct auf Levels",
        "seasonal_naive": "Seasonal Naive",
        "recursive_logdiff": "Rekursiv auf Log-Diffs",
        "direct_logdiff": "Direkt auf Log-Diffs",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for name in ["direct_level", "seasonal_naive", "recursive_logdiff", "direct_logdiff"]:
        hs = sorted(summary[name], key=int)
        axes[0].plot(hs, [summary[name][h]["mae"] for h in hs], marker="o",
                     color=colors[name], label=labels[name])
        axes[1].plot(hs, [summary[name][h]["dir_acc"] for h in hs], marker="o",
                     color=colors[name], label=labels[name])
    axes[0].set_yscale("log")
    axes[0].set_title("MAE auf Levels (log-Skala)")
    axes[1].set_title("Directional Accuracy")
    axes[1].axhline(0.5, color="#888888", ls=":", lw=1)
    for ax in axes:
        ax.set_xlabel("Horizont (Monate)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "E6: Level- vs. Log-Differenz-Prognosen - direkte Formulierung schlaegt Rekursion",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "e6_levels_vs_logdiff")

    save_result("e6_levels_vs_logdiff", {"metrics_on_levels": summary})
    return {"summary": summary}


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
