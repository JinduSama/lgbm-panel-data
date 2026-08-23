"""
E6 - Level- vs. Log-Differenz-Prognosen ueber Trend-Regime.

Fragestellung: Sind direkte Level-Forecasts besser als Log-Differenz-
Forecasts - und was kostet die Rekursivitaet der Differenzen? Und: Wie
aendert sich die Antwort mit Form und Richtung des Trends?

Fuenf Trend-Regime, jeweils vier Varianten:
    direct_level     : ein LGBM pro Horizont auf rohen Levels
    seasonal_naive   : Referenz-Baseline
    direct_logdiff   : ein LGBM pro Horizont auf der h-Schritt-Log-Aenderung
                       log y[t+h] - log y[t]; Rekonstruktion aus dem
                       beobachteten Anker exp(log y[t] + pred)
    recursive_logdiff: EIN Modell fuer 1-Schritt-Aenderungen, Rollout mit
                       eigenen Prognosen als Lags

Regime:
    kein_trend       : growth = 0                     (stationaer + Saison)
    leicht_trendend  : growth 0.2-0.5 %/Monat         (~2.4-6 %/Jahr)
    linear_trendend  : additiv +2..6 Einheiten/Monat
    stark_trendend   : growth 1.2-3 %/Monat           (wie alte E6-Version)
    trendumkehr      : +1.5-2.5 %/Monat bis Monat 96,
                       danach -60 % der Wachstumsrate (Strukturbruch)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import save_fig, save_result

from lgbm_panel.experiments import ModelSpec, expanding_backtest
from lgbm_panel.features import FeatureConfig, build_supervised
from lgbm_panel.strategies import DirectLGBM

HZ = (1, 3, 6, 12, 18)
N_FOLDS = 3
N_SERIES = 60
N_PERIODS = 144
# Explizit ohne Exogena: verhindert, dass Hilfsspalten als Features einschleppen.
DIRECT_CFG = FeatureConfig(exog_cols=())

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

REGIMES: dict[str, dict] = {
    "kein_trend": {"kind": "exp", "growth": (0.0, 0.0)},
    "leicht_trendend": {"kind": "exp", "growth": (0.002, 0.005)},
    "linear_trendend": {"kind": "linear", "slope": (2.0, 6.0)},
    "stark_trendend": {"kind": "exp", "growth": (0.012, 0.03)},
    "trendumkehr": {"kind": "reversal", "growth": (0.015, 0.025), "switch": 96},
}


def _panel(regime: dict, seed: int) -> pd.DataFrame:
    """
    Panel fuer ein Regime (lokal gebaut, Spiegel von make_panel).

    Bewusst hohe Basis-Level (>= 80): Saison (bis 35 abs.) und Rauschen
    duerfen nie negative Werte erzeugen - sonst produziert der Log
    Clipping-Spikes und die Log-Differenz-VariantenMuell-Labels.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=N_PERIODS, freq="MS")
    frames = []
    for s in range(N_SERIES):
        t = np.arange(N_PERIODS, dtype=float)
        amp = rng.uniform(15.0, 35.0)
        phase = rng.uniform(0, 2 * np.pi)
        eps = np.zeros(N_PERIODS)
        shocks = rng.normal(0, rng.uniform(2.0, 5.0), N_PERIODS)
        for i in range(1, N_PERIODS):
            eps[i] = 0.3 * eps[i - 1] + shocks[i]
        season = amp * np.sin(2 * np.pi * t / 12.0 + phase)

        if regime["kind"] == "linear":
            level = rng.uniform(150, 400)
            slope = rng.uniform(*regime["slope"])
            y = level + slope * t + season + eps
        elif regime["kind"] == "reversal":
            # Wachstum kippt am Stichtag ins Negative (-60 % der Rate).
            level = rng.uniform(120, 350)
            g_up = rng.uniform(*regime["growth"])
            g = np.where(t < regime["switch"], g_up, -0.6 * g_up)
            y = level * np.exp(np.cumsum(g)) + season + eps
        else:
            level = rng.uniform(80, 400)
            growth = rng.uniform(*regime["growth"])
            y = level * np.exp(growth * t) + season + eps

        frames.append(pd.DataFrame({"series": f"S{s:03d}", "date": dates, "value": y}))
    return pd.concat(frames, ignore_index=True)


def _fold_ends(grid: pd.DatetimeIndex, step: int) -> list[pd.Timestamp]:
    """Identische Fold-Grenzen wie expanding_backtest."""
    return [
        grid[-1] - pd.DateOffset(months=(N_FOLDS - k + 1) * step)
        for k in range(1, N_FOLDS + 1)
    ]


def _row_from_history(hist: np.ndarray, date: pd.Timestamp) -> dict[str, float]:
    """Feature-Zeile in build_supervised-Semantik von RCFG."""
    row: dict[str, float] = {}
    for k in R_LAGS:
        row[f"lag_{k}"] = hist[-k]
    for w in R_WINDOWS:
        row[f"roll{w}_mean"] = float(np.mean(hist[-w:]))
    row["month"] = date.month
    return row


def _run_regime(raw: pd.DataFrame) -> dict[str, dict]:
    """Vier-Varianten-Vergleich auf einem Panel; Metriken auf Levels."""
    log_df = raw.assign(value=np.log(raw["value"].clip(lower=1e-6)))
    grid = pd.DatetimeIndex(sorted(raw["date"].unique()))
    ends = _fold_ends(grid, max(HZ))
    level_lookup = raw.set_index(["series", "date"])["value"]

    # 1) Engine-Modelle: Levels + Baseline.
    engine = expanding_backtest(
        raw,
        horizons=HZ,
        specs=[ModelSpec("direct_level"), ModelSpec("seasonal_naive", kind="snaive")],
        n_folds=N_FOLDS,
        step_months=max(HZ),
    )
    summary: dict[str, dict] = {
        m: {
            str(int(h)): row
            for h, row in grp.set_index("horizon")
            [["mae", "rmse", "smape", "dir_acc"]].to_dict("index").items()
        }
        for m, grp in engine.metrics_by_horizon.groupby("model")
    }
    summary.pop("naive", None)

    # 2) Direkt auf Log-Diffs.
    sup = build_supervised(log_df, horizons=HZ, config=DIRECT_CFG).merge(
        log_df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
        on=["series", "date"],
        how="left",
    )
    sup["y_change"] = sup["y"] - sup["y_ref"]
    direct_rows = []
    for i, fe in enumerate(ends, start=1):
        hi = fe + pd.DateOffset(months=max(HZ))
        train = sup[sup["target_date"] <= fe].dropna(subset=["y_change"])
        test = sup[(sup["target_date"] > fe) & (sup["target_date"] <= hi)]
        if train.empty or test.empty:
            continue
        model = DirectLGBM(horizons=HZ).fit(
            train.assign(y=train["y_change"]), config=DIRECT_CFG, num_boost_round=300
        )
        p = model.predict(test)
        p["level_pred"] = np.exp(p["y_ref"] + p["pred"])
        p["truth"] = np.exp(p["y"])  # y ist hier das Log-Level zum Ziel.
        p["L0"] = np.exp(p["y_ref"])
        p["fold"] = i
        direct_rows.append(p)
    summary.update(_evaluate(pd.concat(direct_rows, ignore_index=True), "direct_logdiff"))

    # 3) Rekursiv auf Log-Diffs.
    sup1 = build_supervised(log_df, horizons=(1,), config=RCFG).merge(
        log_df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
        on=["series", "date"],
        how="left",
    )
    sup1 = sup1.dropna(subset=["y_ref"])
    rec_model = DirectLGBM(horizons=(1,), categorical=()).fit(
        sup1.assign(y=sup1["y"] - sup1["y_ref"]), config=RCFG, num_boost_round=300
    )
    feat_cols = list(_row_from_history(np.zeros(max(R_LAGS)), pd.Timestamp("2020-01-01")))
    log_hist = {
        s: g.sort_values("date")["value"].to_numpy()
        for s, g in log_df.groupby("series", sort=False)
    }
    recursive_rows = []
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
                    "L0": float(np.exp(v_end)),
                    "fold": i,
                })
    summary.update(_evaluate(pd.DataFrame(recursive_rows), "recursive_logdiff"))
    return summary


def _evaluate(preds: pd.DataFrame, name: str) -> dict[str, dict]:
    out: dict[str, dict] = {name: {}}
    for h, grp in preds.groupby("horizon"):
        ok = grp.dropna(subset=["level_pred", "truth"])
        out[name][str(int(h))] = {
            "mae": float(np.mean(np.abs(ok["truth"] - ok["level_pred"]))),
            "rmse": float(np.sqrt(np.mean((ok["truth"] - ok["level_pred"]) ** 2))),
            "dir_acc": float(
                np.mean(
                    np.sign(ok["level_pred"] - ok["L0"])
                    == np.sign(ok["truth"] - ok["L0"])
                )
            ),
        }
    return out


def run() -> dict:
    all_results: dict[str, dict] = {}

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    axes = axes.ravel()
    colors = {
        "direct_level": "#9d9d9d",
        "seasonal_naive": "#f4a261",
        "recursive_logdiff": "#d1495b",
        "direct_logdiff": "#00798c",
    }
    labels = {
        "direct_level": "Direkt auf Levels",
        "seasonal_naive": "Seasonal Naive",
        "recursive_logdiff": "Rekursiv auf Log-Diffs",
        "direct_logdiff": "Direkt auf Log-Diffs",
    }

    for idx, (name, regime) in enumerate(REGIMES.items()):
        raw = _panel(regime, seed=33)
        summary = _run_regime(raw)
        all_results[name] = summary

        ax = axes[idx]
        for model in [
            "direct_level", "seasonal_naive",
            "recursive_logdiff", "direct_logdiff",
        ]:
            pts = sorted((int(h), m["mae"]) for h, m in summary[model].items())
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                marker="o", color=colors[model], label=labels[model],
            )
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("Horizont (Monate)", fontsize=9)

    for ax in axes[len(REGIMES):]:
        ax.axis("off")
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "E6: Level- vs. Log-Differenz-Prognosen ueber Trend-Regime (MAE, log-Skala)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_fig(fig, "e6_levels_vs_logdiff")

    save_result("e6_levels_vs_logdiff", {"metrics_on_levels": all_results})
    return {"regimes": all_results}


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
