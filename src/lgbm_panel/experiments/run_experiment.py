"""
Expanding-Window-Backtest für LGBM Panel-Forecasting.

Leakage-Regel: Eine Trainingszeile (Cutoff t, Ziel t+h) darf nur verwendet
werden, wenn das Ziel bereits beobachtet ist -> ``target_date <= fold_end``.
Die Testzeilen eines Folds haben Ziele strikt nach ``fold_end``.

Baseline-Modelle (Naive / SeasonalNaive) werden analytisch aus dem Roh-Panel
berechnet (SeasonalNaive: Wert exakt 12 Monate vor dem Zielmonat), nicht aus
den Lag-Spalten der supervised-Tabelle.

Verwendung:
    from lgbm_panel.experiments import expanding_backtest, ModelSpec
    res = expanding_backtest(df, horizons=(1, 6, 12, 18), n_folds=4)
    print(res.metrics_by_horizon)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.build_features import FeatureConfig, build_supervised
from ..metrics import directional_accuracy, mae, rmse, smape
from ..strategies.direct_forecast import DirectLGBM


# --------------------------------------------------------------------------- #
# Modell-Spezifikationen
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """Ein zu evaluierendes Modell innerhalb des Backtests."""

    name: str
    kind: str = "lgbm"  # "lgbm" | "naive" | "snaive"
    config: FeatureConfig | None = None
    num_boost_round: int = 400
    categorical: tuple[str, ...] = ("series",)


@dataclass
class BacktestResult:
    """Ergebnis eines Expanding-Window-Backtests."""

    predictions: pd.DataFrame  # model, fold, series, cutoff, target_date, horizon, y_ref, y, pred
    metrics_by_horizon: pd.DataFrame  # model x horizon -> MAE/RMSE/sMAPE/DirAcc (Fold-Mittel)
    fold_metrics: pd.DataFrame  # model x fold x horizon
    importance: pd.DataFrame | None = None  # gain-Importance je Fold (nur lgbm-Specs)


def default_specs() -> list[ModelSpec]:
    """Default-Vergleich: globales LGBM gegen die beiden Naiv-Baselines."""
    return [
        ModelSpec(name="lgbm", kind="lgbm"),
        ModelSpec(name="seasonal_naive", kind="snaive"),
        ModelSpec(name="naive", kind="naive"),
    ]


def _config_key(cfg: FeatureConfig) -> tuple:
    return (
        tuple(cfg.lags),
        tuple(cfg.rolling_windows),
        tuple(cfg.rolling_stats),
        tuple(cfg.time_features),
        tuple(cfg.diff_lags),
        cfg.use_cross_sectional,
        tuple(cfg.exog_lags),
        None if cfg.exog_cols is None else tuple(cfg.exog_cols),
    )


def _panel_lookup(df: pd.DataFrame) -> pd.Series:
    """(series, date) -> value Lookup."""
    return df.set_index(["series", "date"])["value"]


def _baseline_predictions(
    df: pd.DataFrame,
    fold_end: pd.Timestamp,
    targets: pd.DatetimeIndex,
    kind: str,
) -> pd.DataFrame:
    """
    Analytische Baselines pro (series, target_date), nur Historie <= fold_end.

    naive  : letzter beobachteter Wert.
    snaive : Wert genau 12 Monate vor dem Zielmonat (falls beobachtet).
    """
    hist = df[df["date"] <= fold_end]
    idx = pd.MultiIndex.from_product(
        [hist["series"].unique(), targets], names=["series", "target_date"]
    )
    out = idx.to_frame(index=False)

    if kind == "naive":
        last = hist.sort_values("date").groupby("series")["value"].last()
        out["pred"] = out["series"].map(last)
    elif kind == "snaive":
        lookup = _panel_lookup(hist)
        shifted = out["target_date"] - pd.DateOffset(months=12)
        out["pred"] = lookup.reindex(pd.MultiIndex.from_arrays([out["series"], shifted])).to_numpy()
    else:
        raise ValueError(f"Unbekannte Baseline: {kind!r}")
    return out


def _eval_block(pred: pd.DataFrame) -> dict[str, float]:
    """Metriken für einen Vorhersage-Block mit Spalten y/pred/y_ref."""
    ok = pred.dropna(subset=["pred", "y"])
    ref = ok["y_ref"] if "y_ref" in ok.columns else None
    if ok.empty:
        return {"mae": np.nan, "rmse": np.nan, "smape": np.nan, "dir_acc": np.nan}
    return {
        "mae": mae(ok["y"], ok["pred"]),
        "rmse": rmse(ok["y"], ok["pred"]),
        "smape": smape(ok["y"], ok["pred"]),
        "dir_acc": directional_accuracy(ok["y"], ok["pred"], y_ref=ref),
    }


def expanding_backtest(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 6, 12, 18),
    specs: list[ModelSpec] | None = None,
    n_folds: int = 4,
    step_months: int | None = None,
    collect_importance: bool = False,
    verbose: bool = False,
) -> BacktestResult:
    """
    Expanding-Window-Backtest ueber ein Long-Format-Panel.

    Parameters
    ----------
    df : Panel mit ``series``, ``date``, ``value`` (+ optionale exogene Spalten).
    horizons : Zu evaluierende Forecast-Horizonte.
    specs : Zu vergleichende Modelle (Default: LGBM + beide Baselines).
    n_folds : Anzahl der Folds; Fold k trainiert auf allen Zielen bis
              ``last - (n_folds-k)*step`` und testet auf den folgenden ``step`` Monaten.
    step_months : Fold-Abstand in Monaten (Default: max(horizons)).
    collect_importance : Gain-Importance der LGBM-Modelle mitsammeln.
    verbose : Fortschritt ausgeben.
    """
    specs = specs or default_specs()
    step = step_months or max(horizons)

    # Supervised-Tabellen einmal je eindeutiger FeatureConfig bauen.
    sup_cache: dict[tuple, pd.DataFrame] = {}
    for spec in specs:
        if spec.kind != "lgbm":
            continue
        key = _config_key(spec.config or FeatureConfig())
        if key not in sup_cache:
            sup_cache[key] = build_supervised(df, horizons=horizons, config=spec.config).merge(
                df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
                on=["series", "date"],
                how="left",
            )

    grid = pd.DatetimeIndex(sorted(df["date"].unique()))
    # Letzter Fold testet genau die letzten ``step`` Monate des Panels.
    ends = [
        grid[-1] - pd.DateOffset(months=(n_folds - k + 1) * step) for k in range(1, n_folds + 1)
    ]

    pred_frames: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    imp_rows: list[pd.DataFrame] = []

    for i, fold_end in enumerate(ends, start=1):
        hi = fold_end + pd.DateOffset(months=step)
        target_window = grid[(grid > fold_end) & (grid <= hi)]

        for spec in specs:
            if spec.kind == "lgbm":
                sup = sup_cache[_config_key(spec.config or FeatureConfig())]
                train = sup[sup["target_date"] <= fold_end]
                test = sup[(sup["target_date"] > fold_end) & (sup["target_date"] <= hi)]
                if train.empty or test.empty:
                    continue
                model = DirectLGBM(horizons=horizons, categorical=spec.categorical)
                model.fit(train, config=spec.config, num_boost_round=spec.num_boost_round)
                out = model.predict(test)
                if collect_importance:
                    for h, booster in model.models.items():
                        imp_rows.append(
                            pd.DataFrame(
                                {
                                    "model": spec.name,
                                    "fold": i,
                                    "horizon": h,
                                    "feature": booster.feature_name(),
                                    "gain": booster.feature_importance("gain"),
                                }
                            )
                        )
            else:
                base = _baseline_predictions(df, fold_end, target_window, spec.kind)
                lookup = _panel_lookup(df)
                parts = []
                for h in horizons:
                    b = base.copy()
                    b["horizon"] = h
                    b["cutoff"] = b["target_date"] - pd.DateOffset(months=h)
                    b["y"] = lookup.reindex(
                        pd.MultiIndex.from_arrays([b["series"], b["target_date"]])
                    ).to_numpy()
                    b["y_ref"] = lookup.reindex(
                        pd.MultiIndex.from_arrays([b["series"], b["cutoff"]])
                    ).to_numpy()
                    parts.append(b[b["cutoff"].notna()])
                out = pd.concat(parts, ignore_index=True)[
                    ["series", "cutoff", "target_date", "horizon", "y_ref", "y", "pred"]
                ]

            out = out.rename(columns={"date": "cutoff"}).assign(model=spec.name, fold=i)
            cols = [
                "model",
                "fold",
                "series",
                "cutoff",
                "target_date",
                "horizon",
                "y_ref",
                "y",
                "pred",
            ]
            out = out[[c for c in cols if c in out.columns]]
            pred_frames.append(out)

            for h, grp in out.groupby("horizon"):
                m = _eval_block(grp)
                fold_rows.append({"model": spec.name, "fold": i, "horizon": int(h), **m})
            if verbose:
                print(
                    f"fold {i}/{len(ends)} end={fold_end.date()} spec={spec.name} rows={len(out)}"
                )

    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    fold_metrics = pd.DataFrame(fold_rows)
    if fold_metrics.empty:
        metrics_by_horizon = fold_metrics
    else:
        metrics_by_horizon = (
            fold_metrics.groupby(["model", "horizon"], as_index=False)
            .mean(numeric_only=True)
            .drop(columns=["fold"])
        )
    importance = pd.concat(imp_rows, ignore_index=True) if imp_rows else None
    return BacktestResult(predictions, metrics_by_horizon, fold_metrics, importance)


if __name__ == "__main__":
    from ..data.load import load_dataset

    panel = load_dataset("synthetic", n_series=12, n_periods=132)
    res = expanding_backtest(panel, horizons=(1, 6, 12, 18), n_folds=3, verbose=True)
    print(res.metrics_by_horizon.pivot(index="horizon", columns="model", values="mae"))
