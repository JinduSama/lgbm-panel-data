"""
E9 - Bringt Hyperparameter-Tuning was? Optuna ueber Szenarien hinweg.

Fragestellung: LGBM kommt mit vernuenftigen Defaults (lr 0.05, 31 Leaves,
subsample 0.9, colsample 0.8, reg_lambda 1). Wie viel MAE gewinnt man,
wenn man die Kern-Hyperparameter pro Datensatz optimiert - und ist die
Antwort ueber Panel/TS-Szenarien hinweg universell?

Szenarien (decken das Spektrum aus E1-E8 ab):
    kein_trend      : stationaer + starke Saison (E6-DGP)
    stark_trendend  : exponentiell +1.2-3 %/Monat (E6-DGP)
    trendumkehr     : Strukturbruch bei Monat 96 (E6-DGP)
    exog_treiber    : Trend + Saison + AR(1)-Treiber x (E8-Welt)
    m4_real         : 150 echte M4-Monatsserien

Protokoll (zeitlich sauber):
    - Modell: globales direktes LGBM auf Levels, identisches Feature-Set
      je Szenario (beim Treiber-Szenario inkl. x).
    - Tuning-Objektiv: gepoolte MAE ueber Horizonte 1/3/6/12/18 auf dem
      VORletzten 18-Monats-Block (Fold 1). TPE, 40 Trials, fester Seed.
    - Bewertung: DEFAULT-Params vs. beste Params auf dem LETZten Block
      (Fold 2) - den sieht die Suche nie.

Suchraum: learning_rate, num_leaves, min_child_samples,
feature_fraction, bagging_fraction, reg_lambda, reg_alpha.
"""

from __future__ import annotations

import e6_levels_vs_logdiff as e6
import e8_combined as e8
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from _common import save_fig, save_result

from lgbm_panel.data import load_dataset
from lgbm_panel.features import FeatureConfig, build_supervised
from lgbm_panel.strategies import DirectLGBM

HZ = (1, 3, 6, 12, 18)
N_TRIALS = 40
N_PERIODS = 144
SEED = 9

SCENARIOS = ["kein_trend", "stark_trendend", "trendumkehr", "exog_treiber", "m4_real"]


def make_cfg(exog: bool) -> FeatureConfig:
    return FeatureConfig(
        lags=(1, 2, 3, 6, 12),
        rolling_windows=(3, 12),
        rolling_stats=("mean",),
        diff_lags=(),
        time_features=("month",),
        exog_cols=("x",) if exog else (),
    )


def _scenario_data(name: str) -> tuple[pd.DataFrame, FeatureConfig]:
    if name == "exog_treiber":
        return e8.make_world(seed=54), make_cfg(True)
    if name == "m4_real":
        return load_dataset("m4", n_series=150), make_cfg(False)
    regime = {
        "kein_trend": e6.REGIMES["kein_trend"],
        "stark_trendend": e6.REGIMES["stark_trendend"],
        "trendumkehr": e6.REGIMES["trendumkehr"],
    }[name]
    return e6._panel(
        regime,
        seed={  # type: ignore[arg-type]
            "kein_trend": 61,
            "stark_trendend": 62,
            "trendumkehr": 63,
        }[name],
    ), make_cfg(False)


def _fold_ends(grid: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Zwei Folds a 18 Monate: [-36,-18] zum Suchen, [-18,0] zum Bewerten."""
    return [grid[-1] - pd.DateOffset(months=(3 - k) * max(HZ)) for k in range(1, 3)]


def _prepare(raw: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    sup = build_supervised(raw, horizons=HZ, config=cfg)
    return sup.dropna(subset=["y"])


def _block_mae(
    sup: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cfg: FeatureConfig, params: dict
) -> float:
    """Gepoolte MAE (alle Horizonte) fuer einen Test-Block ab einem Fit bis `start`."""
    train = sup[sup["target_date"] <= start]
    test = sup[(sup["target_date"] > start) & (sup["target_date"] <= end)]
    model = DirectLGBM(horizons=HZ).fit(train, config=cfg, num_boost_round=300, **params)
    p = model.predict(test)
    return float(np.mean(np.abs(p["y"] - p["pred"])))


def tune_scenario(name: str) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    raw, cfg = _scenario_data(name)
    sup = _prepare(raw, cfg)
    grid = pd.DatetimeIndex(sorted(raw["date"].unique()))
    fe_search, fe_eval = _fold_ends(grid)
    hi_search = fe_search + pd.DateOffset(months=max(HZ))
    hi_eval = fe_eval + pd.DateOffset(months=max(HZ))

    mae_default = _block_mae(sup, fe_eval, hi_eval, cfg, {})

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "colsample_bytree": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "subsample": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        return _block_mae(sup, fe_search, hi_search, cfg, params)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = {k: v for k, v in study.best_params.items()}
    mapped = {
        "learning_rate": best["learning_rate"],
        "num_leaves": int(best["num_leaves"]),
        "min_child_samples": int(best["min_child_samples"]),
        "colsample_bytree": best["feature_fraction"],
        "subsample": best["bagging_fraction"],
        "reg_lambda": best["reg_lambda"],
        "reg_alpha": best["reg_alpha"],
    }
    mae_tuned = _block_mae(sup, fe_eval, hi_eval, cfg, mapped)

    return {
        "mae_holdout_default": mae_default,
        "mae_holdout_tuned": mae_tuned,
        "improvement_pct": 100.0 * (1.0 - mae_tuned / mae_default),
        "best_trial_value": float(study.best_value),
        "best_params": mapped,
    }


STYLE = "#2c7fb8"


def run() -> dict:
    results = {name: tune_scenario(name) for name in SCENARIOS}

    # ---------------- Abbildung -------------------------------------------
    fig, (ax_pooled, ax_gain) = plt.subplots(1, 2, figsize=(13.5, 5))
    xs = np.arange(len(SCENARIOS))
    defaults = [results[s]["mae_holdout_default"] for s in SCENARIOS]
    tuned = [results[s]["mae_holdout_tuned"] for s in SCENARIOS]
    ax_pooled.bar(xs - 0.2, defaults, width=0.38, color="#9d9d9d", label="Defaults")
    ax_pooled.bar(xs + 0.2, tuned, width=0.38, color=STYLE, label="Optuna (40 Trials)")
    for i, s in enumerate(SCENARIOS):
        imp = results[s]["improvement_pct"]
        ax_pooled.text(
            xs[i],
            max(defaults[i], tuned[i]),
            f"{'-' if imp >= 0 else '+'}{abs(imp):.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1b4943" if imp >= 0 else "#b23a48",
        )
    ax_pooled.set_xticks(xs)
    ax_pooled.set_xticklabels(SCENARIOS, rotation=20, ha="right")
    ax_pooled.set_ylabel(f"Gepoolter MAE (h={'/'.join(map(str, HZ))})")
    ax_pooled.set_title("E9: Defaults vs. getunte Params je Szenario")
    ax_pooled.legend(fontsize=8, frameon=False)
    ax_pooled.grid(alpha=0.25, axis="y")

    gains = [results[s]["improvement_pct"] for s in SCENARIOS]
    colors = ["#1b4943" if g >= 0 else "#b23a48" for g in gains]
    ax_gain.bar(xs, gains, width=0.55, color=colors)
    ax_gain.axhline(0, color="#666666", lw=0.8)
    ax_gain.set_xticks(xs)
    ax_gain.set_xticklabels(SCENARIOS, rotation=20, ha="right")
    ax_gain.set_ylabel("MAE-Verbesserung (%)")
    ax_gain.set_title("Was bringt Tuning wirklich?")
    ax_gain.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    save_fig(fig, "e9_tuning")

    save_result("e9_tuning", {"scenarios": results})
    return {"scenarios": results}


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
