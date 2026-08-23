"""
E4 - Kausale Plausibilitaet: Vorhersage != Erklaerung.

DGP mit persistentem exogenem Treiber (z.B. Marketing-Budget):
    x_t ~ AR(1), phi=0.95            (kausaler Treiber von y)
    y_t = level + beta * x_{t-1} + Saison + Rauschen

Ablauf:
1. Regime-Vergleich (honest Backtest vor der Intervention):
   Lag-only-Modell und Treiber-Modell sind aehnlich gut -> anhand der
   Proguosequalitaet allein ist der kausal richtige Modell nicht erkennbar.
2. Intervention: ab Monat T wird x auf 35% gesenkt (do-Operator). Die
   Counterfactual-Welt ist bekannt, weil das DGP synthetisch ist.
3. Prognosen ab Origin T fuer 18 Monate:
   - M_lag   : nur Target-Lags        (korrelierte Vergangenheit)
   - M_x     : + aktueller x-Stand    (sieht die Senkung nicht)
   - M_x_plan: + geplanter Pfad x[target-1] ("Was-waere-wenn" mit bekannten
               Budgetplaenen)         (trackt die Senkung)

Erkenntnis-Ziele:
- Lag-Modelle extrapolieren das alte Regime: systematisch falsch nach Eingriff.
- Nur ein Modell mit dem kausalen Treiber (+ Szenario-Pfad) reagiert korrekt.
- Gain-Importance zeigt den Unterschied: Lags vs. Treiber.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import metrics_dict, save_fig, save_result
from lgbm_panel.experiments import ModelSpec, expanding_backtest
from lgbm_panel.features import FeatureConfig, build_supervised
from lgbm_panel.strategies import DirectLGBM

HORIZONS = tuple(range(1, 19))
N_PERIODS = 150
T_INT = 132  # letzter Monat vor der Intervention
CUT = 0.35


def make_world(n_series: int = 40, seed: int = 11):
    """Faktische und Counterfactual-Welt (x ab T auf CUT gesenkt)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=N_PERIODS, freq="MS")
    fact_frames, cf_frames = [], []

    for s in range(n_series):
        level = rng.uniform(120, 220)
        beta = rng.uniform(2.0, 3.0)
        amp = rng.uniform(8.0, 18.0)
        phase = rng.uniform(0, 2 * np.pi)
        x_fact = np.zeros(N_PERIODS)
        # OU-artiger Prozess um Level 45: bleibend positiv und persistent.
        x_fact[0] = 45.0
        shocks = rng.normal(0, 2.5, N_PERIODS)
        for i in range(1, N_PERIODS):
            x_fact[i] = 45.0 + 0.9 * (x_fact[i - 1] - 45.0) + shocks[i]
        noise = rng.normal(0, 6.0, N_PERIODS)

        def dgp(x, _level=level, _beta=beta, _amp=amp, _phase=phase, _noise=noise):
            t = np.arange(N_PERIODS, dtype=float)
            season = _amp * np.sin(2 * np.pi * t / 12.0 + _phase)
            return _level + _beta * np.roll(x, 1) + season + _noise

        x_cf = x_fact.copy()
        x_cf[T_INT:] = x_fact[T_INT:] * CUT

        fact_frames.append(
            pd.DataFrame({"series": f"S{s:02d}", "date": dates, "value": dgp(x_fact), "x": x_fact})
        )
        cf_frames.append(
            pd.DataFrame({"series": f"S{s:02d}", "date": dates, "value": dgp(x_cf), "x": x_cf})
        )

    return pd.concat(fact_frames, ignore_index=True), pd.concat(cf_frames, ignore_index=True)


LAG_ONLY = FeatureConfig(exog_cols=())
WITH_X = FeatureConfig(exog_cols=("x",))
WITH_X_PLAN = FeatureConfig(exog_cols=("x",), exog_scenario_lags=(1,))


def _forecast_from_origin(
    model: DirectLGBM,
    df: pd.DataFrame,
    cfg: FeatureConfig,
    origin: pd.Timestamp,
    plan_path: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Zeilen am Cutoff=origin je Serie und Horizont bauen und prognostizieren.

    Trick: Panel um 18 Monate mit Dummy-Werten verlaengern, damit
    ``build_supervised`` die Origin-Zeilen behaelt (y dort Dummy).
    ``plan_path`` ersetzt x_at_tminus1 durch den geplanten Pfad.
    """
    last = df["date"].max()
    pad_months = max(HORIZONS)
    pad_dates = pd.date_range(last + pd.DateOffset(months=1), periods=pad_months, freq="MS")
    last_vals = df.sort_values("date").groupby("series").tail(1).set_index("series")[["value", "x"]]
    pads = []
    for s, row in last_vals.iterrows():
        p = pd.DataFrame({"date": pad_dates})
        p.insert(0, "series", s)
        p["value"] = row["value"]
        if "x" in df.columns:
            p["x"] = row["x"]
        pads.append(p)
    padded = pd.concat([df, *pads], ignore_index=True)

    sup = build_supervised(padded, horizons=HORIZONS, config=cfg)
    rows = sup[sup["date"] == origin].copy()
    if plan_path is not None:
        lookup = plan_path.set_index(["series", "date"])["x"]
        key = pd.MultiIndex.from_arrays(
            [rows["series"], rows["target_date"] - pd.DateOffset(months=1)]
        )
        rows["x_at_tminus1"] = lookup.reindex(key).to_numpy()
    out = model.predict(rows)
    return out.dropna(subset=["pred"])


def run() -> dict:
    fact, cf = make_world()

    # --- 1) Regime-Vergleich: beide Modelle wirken gleich gut -------------
    pre = fact[fact["date"] < fact["date"].max() - pd.DateOffset(months=18)]
    regime = expanding_backtest(
        pre,
        horizons=(1, 6, 12, 18),
        specs=[
            ModelSpec("lag_only", config=LAG_ONLY),
            ModelSpec("with_x", config=WITH_X),
        ],
        n_folds=3,
        step_months=18,
    )

    train_end = fact["date"].unique()[T_INT - 1]
    train_raw = fact[fact["date"] <= train_end]
    models = {}
    for name, cfg in (
        ("lag_only", LAG_ONLY),
        ("with_x", WITH_X),
        ("with_x_plan", WITH_X_PLAN),
    ):
        train = build_supervised(train_raw, horizons=HORIZONS, config=cfg)
        models[name] = DirectLGBM(horizons=HORIZONS).fit(train, config=cfg, num_boost_round=300)

    # --- 3) Prognosen ab Origin unter der Intervention ---------------------
    origin = train_end
    preds = {
        "lag_only": _forecast_from_origin(models["lag_only"], train_raw, LAG_ONLY, origin, None),
        "with_x": _forecast_from_origin(models["with_x"], train_raw, WITH_X, origin, None),
        "with_x_plan": _forecast_from_origin(
            models["with_x_plan"], train_raw, WITH_X_PLAN, origin, cf
        ),
    }

    truth = cf[(cf["date"] > origin)]
    truth_lookup = truth.set_index(["series", "date"])["value"]
    eval_rows = []
    for name, p in preds.items():
        p = p.copy()
        p["truth"] = truth_lookup.reindex(
            pd.MultiIndex.from_arrays([p["series"], p["target_date"]])
        ).to_numpy()
        p = p.dropna(subset=["truth"])
        ref = (
            train_raw.set_index(["series", "date"])["value"]
            .reindex(pd.MultiIndex.from_arrays([p["series"], p["date"]]))
            .to_numpy()
        )
        eval_rows.append(
            {
                "model": name,
                "mae": float(np.mean(np.abs(p["truth"] - p["pred"]))),
                "bias": float(np.mean(p["pred"] - p["truth"])),
                "dir_acc": float(np.mean(np.sign(p["pred"] - ref) == np.sign(p["truth"] - ref))),
            }
        )
    intervention = pd.DataFrame(eval_rows)

    # --- Importance-Kontrast ------------------------------------------------
    def gain_shares(cfg: FeatureConfig) -> dict[str, float]:
        sup = build_supervised(train_raw, horizons=(12,), config=cfg)
        m = DirectLGBM(horizons=(12,)).fit(sup, config=cfg, num_boost_round=300)
        booster = m.models[12]
        names = booster.feature_name()
        gains = booster.feature_importance("gain")
        fam = {}
        for n_, g_ in zip(names, gains, strict=True):
            key = (
                "Target-Lags"
                if n_.startswith(("lag_", "diff_"))
                else "Rolling"
                if n_.startswith("roll")
                else "Kalender"
                if n_ in ("month", "quarter", "year")
                else "Treiber x"
                if n_.startswith("x")
                else n_
            )
            fam[key] = fam.get(key, 0.0) + float(g_)
        total = sum(fam.values())
        return {k: round(v / total, 4) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])}

    shares_lag = gain_shares(LAG_ONLY)
    shares_x = gain_shares(WITH_X_PLAN)

    # --- Figur ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    one = "S00"
    hist = fact[fact["series"] == one].sort_values("date")
    hist_cf = cf[(cf["series"] == one) & (cf["date"] > origin)].sort_values("date")
    hist_fact_future = fact[(fact["series"] == one) & (fact["date"] > origin)].sort_values("date")
    ax.plot(hist["date"], hist["value"], color="#333333", lw=1, label="Historie (faktisch)")
    ax.plot(
        hist_fact_future["date"],
        hist_fact_future["value"],
        color="#999999",
        ls="--",
        lw=1,
        label="ohne Intervention (faktisch)",
    )
    ax.plot(
        hist_cf["date"],
        hist_cf["value"],
        color="black",
        lw=2,
        label="nach Budget-Senkung (Wahrheit)",
    )
    colors = {"lag_only": "#d1495b", "with_x": "#edae49", "with_x_plan": "#00798c"}
    for name, p in preds.items():
        pp = p[p["series"] == one].sort_values("target_date")
        ax.plot(pp["target_date"], pp["pred"], color=colors[name], lw=2, label=f"Prognose {name}")
    ax.axvline(origin, color="#666666", ls=":", lw=1)
    ax.set_title(f"Serie {one}: Intervention zum Stichtag")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)

    ax = axes[1]
    for name, p in preds.items():
        merged = p.copy()
        merged["truth"] = truth_lookup.reindex(
            pd.MultiIndex.from_arrays([merged["series"], merged["target_date"]])
        ).to_numpy()
        mae_h = (
            merged.dropna(subset=["truth"])
            .groupby("horizon")
            .apply(
                lambda g_: float(np.mean(np.abs(g_["truth"] - g_["pred"]))), include_groups=False
            )
        )
        ax.plot(mae_h.index, mae_h.values, marker="o", color=colors[name], label=name)
    ax.set_title("MAE nach Horizont (Interventionsfenster)")
    ax.set_xlabel("Horizont (Monate)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    width = 0.38
    keys = sorted(set(shares_lag) | set(shares_x))
    ax.barh(
        [k + " (lag)" for k in keys],
        [shares_lag.get(k, 0) for k in keys],
        height=width,
        color="#d1495b",
        label="lag_only",
    )
    ax.barh(
        [k + " (x)" for k in keys],
        [shares_x.get(k, 0) for k in keys],
        height=width,
        color="#00798c",
        label="with_x_plan",
    )
    ax.set_title("Gain-Anteile (h=12)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("E4: Kausale Plausibilitaet - Intervention im exogenen Treiber", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "e4_causal_intervention")

    payload = {
        "regime_backtest": metrics_dict(regime.metrics_by_horizon),
        "intervention": intervention.to_dict("records"),
        "gain_shares_lag_only_h12": shares_lag,
        "gain_shares_with_x_h12": shares_x,
    }
    save_result("e4_causal", payload)
    return payload


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
