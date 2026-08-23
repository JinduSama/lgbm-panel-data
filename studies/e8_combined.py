"""
E8 - Alles zusammen: Eigenschaften-Konfrontation statt Isolation.

E1-E7 haben Effekte isoliert (je ein DGP pro Frage). Diese Studie kombiniert
alle Eigenschaften in EINEM realistischen DGP und stellt die Ansätze direkt
gegeneinander:

DGP (60 Serien x 144 Monate, wie "normale" Unternehmensdaten):
    moderater Exponentialtrend   growth 0.4-0.9 %/Monat (~5-11 %/Jahr)
    moderate Saison              Amplitude 8-18 absolut
    exogener Treiber x           AR(1), phi=0.9 um 45, Effekt beta*x[t-1],
                                 beta 1.5-2.5 (z.B. Marketing-Budget)
    AR(1)-Rauschen               phi=0.3, sd ~3

2x2-Faktor ueber (Formulierung x Treiber-Info) + Baseline:
    levels      : direktes LGBM auf Levels, nur Target-Lags
    levels_x    : direktes LGBM auf Levels, + x / x_lag1
    logdiff     : direkt auf h-Schritt-Log-Aenderung, ohne x
    logdiff_x   : dito, + x / x_lag1
    seasonal_naive : Referenz

Fragen:
- Addieren sich die Vorteile von Differenzen (Trend-Robustheit) und
  exogenem Treiber (Fuehrungssignal)?
- Wie gross ist der Gain-Anteil von x ueber den Horizont?
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
SEED = 7


def make_cfg(exog: bool) -> FeatureConfig:
    """Identisches Feature-Set fuer alle vier Zellen - nur x unterscheidet."""
    return FeatureConfig(
        lags=(1, 2, 3, 6, 12),
        rolling_windows=(3, 12),
        rolling_stats=("mean",),
        diff_lags=(),
        time_features=("month",),
        exog_cols=("x",) if exog else (),
    )


def make_world(seed: int = SEED) -> pd.DataFrame:
    """Panel mit Trend + Saison + exogenem Treiber + AR-Rauschen."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=N_PERIODS, freq="MS")
    frames = []
    for s in range(N_SERIES):
        t = np.arange(N_PERIODS, dtype=float)
        level0 = rng.uniform(100.0, 300.0)
        growth = rng.uniform(0.004, 0.009)
        amp = rng.uniform(8.0, 18.0)
        phase = rng.uniform(0, 2 * np.pi)
        beta = rng.uniform(1.5, 2.5)

        x = np.zeros(N_PERIODS)
        x[0] = 45.0
        shocks = rng.normal(0, 2.5, N_PERIODS)
        for i in range(1, N_PERIODS):
            x[i] = 45.0 + 0.9 * (x[i - 1] - 45.0) + shocks[i]

        eps = np.zeros(N_PERIODS)
        noise = rng.normal(0, 3.0, N_PERIODS)
        for i in range(1, N_PERIODS):
            eps[i] = 0.3 * eps[i - 1] + noise[i]

        season = amp * np.sin(2 * np.pi * t / 12.0 + phase)
        y = level0 * np.exp(growth * t) + beta * np.roll(x, 1) + season + eps
        frames.append(
            pd.DataFrame({"series": f"S{s:03d}", "date": dates, "value": y, "x": x})
        )
    return pd.concat(frames, ignore_index=True)


def _fold_ends(grid: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return [
        grid[-1] - pd.DateOffset(months=(N_FOLDS - k + 1) * max(HZ))
        for k in range(1, N_FOLDS + 1)
    ]


def _evaluate(preds: pd.DataFrame, name: str) -> dict[str, dict]:
    out: dict[str, dict] = {name: {}}
    for h, grp in preds.groupby("horizon"):
        ok = grp.dropna(subset=["level_pred", "truth"])
        err = ok["truth"] - ok["level_pred"]
        out[name][str(int(h))] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "dir_acc": float(
                np.mean(
                    np.sign(ok["level_pred"] - ok["L0"])
                    == np.sign(ok["truth"] - ok["L0"])
                )
            ),
        }
    return out


def _x_share(model: DirectLGBM) -> dict[str, float]:
    """Gain-Anteil der x-Features je Horizont (letzte Folds mitteln hier: 1 Fit)."""
    shares: dict[str, float] = {}
    for h, booster in model.models.items():
        names = booster.feature_name()
        gains = booster.feature_importance(importance_type="gain")
        total = float(gains.sum())
        xg = float(sum(g for n, g in zip(names, gains, strict=True) if n.startswith("x")))
        shares[str(int(h))] = xg / total if total > 0 else 0.0
    return shares


def run() -> dict:
    raw = make_world()
    log_df = raw.assign(value=np.log(raw["value"].clip(lower=1e-6)))
    grid = pd.DatetimeIndex(sorted(raw["date"].unique()))
    ends = _fold_ends(grid)

    summary: dict[str, dict] = {}
    importance: dict[str, dict[str, float]] = {}

    # 1) Levels +/- x und Seasonal-Naive ueber die Engine.
    engine = expanding_backtest(
        raw,
        horizons=HZ,
        specs=[
            ModelSpec("levels", config=make_cfg(False)),
            ModelSpec("levels_x", config=make_cfg(True)),
            ModelSpec("seasonal_naive", kind="snaive"),
        ],
        n_folds=N_FOLDS,
        step_months=max(HZ),
    )
    summary.update(
        {
            m: {
                str(int(h)): row
                for h, row in grp.set_index("horizon")
                [["mae", "rmse", "smape", "dir_acc"]].to_dict("index").items()
            }
            for m, grp in engine.metrics_by_horizon.groupby("model")
        }
    )
    summary.pop("naive", None)

    # 2) Direkt auf h-Schritt-Log-Diffs, ohne/mit x.
    for tag, exog in (("logdiff", False), ("logdiff_x", True)):
        cfg = make_cfg(exog)
        sup = build_supervised(log_df, horizons=HZ, config=cfg).merge(
            log_df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
            on=["series", "date"],
            how="left",
        )
        sup["y_change"] = sup["y"] - sup["y_ref"]
        rows = []
        for i, fe in enumerate(ends, start=1):
            hi = fe + pd.DateOffset(months=max(HZ))
            train = sup[sup["target_date"] <= fe].dropna(subset=["y_change"])
            test = sup[(sup["target_date"] > fe) & (sup["target_date"] <= hi)]
            if train.empty or test.empty:
                continue
            model = DirectLGBM(horizons=HZ).fit(
                train.assign(y=train["y_change"]), config=cfg, num_boost_round=300
            )
            p = model.predict(test)
            p["level_pred"] = np.exp(p["y_ref"] + p["pred"])
            p["truth"] = np.exp(p["y"])  # y ist das Log-Level zum Ziel.
            p["L0"] = np.exp(p["y_ref"])
            p["fold"] = i
            rows.append(p)
            if i == len(ends):
                importance[tag] = _x_share(model)
        summary.update(_evaluate(pd.concat(rows, ignore_index=True), tag))

    # ---------------- Abbildung: MAE-Kurven + x-Gain-Anteile --------------
    style = {
        "seasonal_naive": ("Seasonal Naive", "#f4a261", "-.", 1.6),
        "levels": ("Levels", "#9d9d9d", "--", 1.8),
        "levels_x": ("Levels + x", "#2c7fb8", "-", 2.0),
        "logdiff": ("Direkt Log-Diff", "#00798c", ":", 2.0),
        "logdiff_x": ("Direkt Log-Diff + x", "#6a4c93", "-", 2.2),
    }
    fig, (ax_mae, ax_gain) = plt.subplots(1, 2, figsize=(13.5, 5))
    for model, (label, color, ls, lw) in style.items():
        hm = summary.get(model, {})
        hs = sorted(int(h) for h in hm)
        ax_mae.plot(hs, [hm[str(h)]["mae"] for h in hs], color=color, ls=ls, lw=lw,
                    marker="o", ms=3.5, label=label)
    ax_mae.set_yscale("log")
    ax_mae.set_xticks(list(HZ))
    ax_mae.set_xlabel("Horizont (Monate)")
    ax_mae.set_ylabel("MAE auf Levels (log)")
    ax_mae.set_title("E8: MAE je Variante - alles kombiniert")
    ax_mae.grid(alpha=0.25)
    ax_mae.legend(fontsize=8, frameon=False)

    width = 0.38
    xs = np.arange(len(HZ))
    for off, (tag, color) in enumerate((("levels_x", "#2c7fb8"), ("logdiff_x", "#6a4c93"))):
        vals = [importance.get(tag, {}).get(str(h), 0.0) for h in HZ]
        ax_gain.bar(xs + (off - 0.5) * width, vals, width=width, color=color,
                    label=f"{style[tag][0]}")
    ax_gain.set_xticks(xs)
    ax_gain.set_xticklabels([str(h) for h in HZ])
    ax_gain.set_ylim(0, 1)
    ax_gain.set_xlabel("Horizont (Monate)")
    ax_gain.set_ylabel("Gain-Anteil von x")
    ax_gain.set_title("Wie viel erklaert der Treiber?")
    ax_gain.grid(alpha=0.25, axis="y")
    ax_gain.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    save_fig(fig, "e8_combined")

    save_result(
        "e8_combined",
        {"metrics_on_levels": summary, "x_gain_share": importance},
    )
    return {"metrics_on_levels": summary, "x_gain_share": importance}


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
