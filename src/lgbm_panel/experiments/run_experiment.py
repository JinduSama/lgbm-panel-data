"""
Expanding-Window-Backtest für LGBM Panel-Forecasting.

Leakage-Regel: Eine Trainingszeile (Cutoff t, Ziel t+h) darf nur verwendet
werden, wenn das Ziel bereits beobachtet ist -> ``target_date <= fold_end``.
Die Testzeilen eines Folds haben Ziele strikt nach ``fold_end``.

Baseline-Modelle (Naive / SeasonalNaive) werden analytisch aus dem Roh-Panel
berechnet, nicht aus den Lag-Spalten der supervised-Tabelle. Der ``origin``
Parameter steuert den Vergleichs-Protokoll:
  "rolling" (Default): Jede Baseline-Zeile nutzt denselben Informationsstand
      wie die LGBM-Zeile mit gleichem (target, horizon): cutoff = target - h.
  "fixed": Klassisches Fixed-Origin-Protokoll (M4) - LGBM testet nur Zeilen
      mit cutoff == fold_end, Baselines nutzen nur Historie <= fold_end.

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
    # Optionale LightGBM-Parameter-Overrides (z.B. objective="quantile").
    params: dict | None = None


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
    targets: pd.DataFrame,
    kind: str,
    horizons: tuple[int, ...],
    origin: str = "rolling",
    fold_ends: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Analytische Baselines je (series, target_date, horizon).

    ``targets`` ist ein Frame mit den Spalten series/target_date (das Fold-
    Fenster je Serie). origin="rolling": Vorhersage aus dem eigenen
    Informationsstand der Zeile (cutoff = target_date - h), exakt wie die
    LGBM-Zeilen:
        naive  : value[cutoff]
        snaive : Wert der letzten Periode mit gleichem Kalendermonat <= cutoff
                 (fuer h <= 12: value[target-12]; darueber -12 iteriert)
    origin="fixed": Blockprognose vom jeweiligen fold_end der Serie;
        naive nutzt value[fold_end], snaive value[target-12k] nur wenn die
        Quelle beobachtet ist (<= fold_end).

    y_ref ist jeweils der letzte am Forecast-Origin beobachtete Wert.
    Zeilen ohne beobachtbaren Vorhersagewert (pred NaN) fallen weg;
    ``_eval_block`` berichtet die Unterlagegroesse als ``n``.
    """
    lookup = _panel_lookup(df)
    base = targets.copy()

    if origin == "fixed":
        if fold_ends is None:
            raise ValueError("origin='fixed' benoetigt fold_ends")
        base["fe"] = base["series"].map(fold_ends)

    parts = []
    for h in horizons:
        b = base.copy()
        b["horizon"] = h
        b["cutoff"] = b["target_date"] - pd.DateOffset(months=h)
        if origin == "fixed":
            # Bucket h = reine h-Schritte-Prognose ab fold_end (je Serie) -
            # dieselben Ziele wie die LGBM-Zeilen im Fixed-Modus.
            b = b[b["target_date"] == b["fe"] + pd.DateOffset(months=h)]
        if kind == "naive":
            # Rolling: Wert am eigenen cutoff; Fixed: Wert am fold_end.
            src = b["cutoff"] if origin == "rolling" else b["fe"]
            b["pred"] = lookup.reindex(
                pd.MultiIndex.from_arrays([b["series"], src])
            ).to_numpy()
        elif kind == "snaive":
            k = -(-h // 12)  # ceil(h/12): gleiche Kalenderperiode, <= Origin
            src = b["target_date"] - pd.DateOffset(months=12 * k)
            pred = lookup.reindex(pd.MultiIndex.from_arrays([b["series"], src])).to_numpy()
            if origin == "fixed":
                # Quelle muss am Origin bereits beobachtet sein.
                pred = np.where(src.to_numpy() <= b["fe"].to_numpy(), pred, np.nan)
            b["pred"] = pred
        else:
            raise ValueError(f"Unbekannte Baseline: {kind!r}")
        ref_src = b["cutoff"] if origin == "rolling" else b["fe"]
        b["y"] = lookup.reindex(
            pd.MultiIndex.from_arrays([b["series"], b["target_date"]])
        ).to_numpy()
        b["y_ref"] = lookup.reindex(
            pd.MultiIndex.from_arrays([b["series"], ref_src])
        ).to_numpy()
        parts.append(b[b["pred"].notna()])
    return pd.concat(parts, ignore_index=True)


def _eval_block(pred: pd.DataFrame) -> dict[str, float]:
    """Metriken für einen Vorhersage-Block mit Spalten y/pred/y_ref."""
    ok = pred.dropna(subset=["pred", "y"])
    ref = ok["y_ref"] if "y_ref" in ok.columns else None
    if ok.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "smape": np.nan, "dir_acc": np.nan}
    return {
        "n": float(len(ok)),
        "mae": mae(ok["y"], ok["pred"]),
        "rmse": rmse(ok["y"], ok["pred"]),
        "smape": smape(ok["y"], ok["pred"]),
        # dir_acc schliesst Zeilen mit undefinierter Richtung aus (d_true == 0
        # oder d_pred == 0); bei rolling-naive ist pred == y_ref -> NaN.
        "dir_acc": directional_accuracy(ok["y"], ok["pred"], y_ref=ref),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Gemeinsame Unterlage + Metriken aus einem Vorhersage-Frame.

    Erwartet die Spalten model/fold/series/target_date/horizon/y_ref/y/pred
    (Schema von ``BacktestResult.predictions``). Schluessel, die nicht von
    JEDEM Modell geliefert werden, fliegen raus ("intersection of non-NaN
    rows"); zurueck geht (gefilterte predictions, metrics_by_horizon,
    fold_metrics).

    Oeffentlich, damit Studien eigene Modell-Arme (z.B. statsforecast,
    per-Serie-Fits) im selben Schema beurteilen und gemeinsam mit den
    expanding_backtest-Armen auswerten koennen.
    """
    if predictions.empty or predictions["model"].nunique() <= 1:
        filtered = predictions.copy()
    else:
        key_cols = ["fold", "horizon", "series", "target_date"]
        ok_keys: set | None = None
        for _, grp in predictions.groupby("model"):
            keys = set(map(tuple, grp[key_cols].itertuples(index=False, name=None)))
            ok_keys = keys if ok_keys is None else (ok_keys & keys)
        all_keys = list(map(tuple, predictions[key_cols].itertuples(index=False, name=None)))
        keep = np.fromiter((k in ok_keys for k in all_keys), dtype=bool, count=len(all_keys))
        filtered = predictions[keep].reset_index(drop=True)

    fold_rows: list[dict] = []
    if not filtered.empty:
        for (model, fold, h), grp in filtered.groupby(["model", "fold", "horizon"]):
            m = _eval_block(grp)
            fold_rows.append({"model": model, "fold": int(fold), "horizon": int(h), **m})
    fold_metrics = pd.DataFrame(fold_rows)
    if fold_metrics.empty:
        return filtered, fold_metrics, fold_metrics
    by_horizon = (
        fold_metrics.groupby(["model", "horizon"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["fold"])
    )
    return filtered, by_horizon, fold_metrics


def per_series_fold_ends(
    df: pd.DataFrame, n_folds: int, step_months: int
) -> dict[int, pd.Series]:
    """
    Fold-Ende je Serie und Fold - dieselbe Verankerung wie expanding_backtest.

    Studien nutzen das z.B. fuer per-Serie-Skalierungen (MASE) konsistent zum
    Backtest-Fenster.
    """
    last = df.groupby("series")["date"].max()
    return {
        k: last - pd.DateOffset(months=(n_folds - k + 1) * step_months)
        for k in range(1, n_folds + 1)
    }


def expanding_backtest(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 6, 12, 18),
    specs: list[ModelSpec] | None = None,
    n_folds: int = 4,
    step_months: int | None = None,
    origin: str = "rolling",
    collect_importance: bool = False,
    verbose: bool = False,
) -> BacktestResult:
    """
    Expanding-Window-Backtest ueber ein Long-Format-Panel.

    Die Folds sind JE SERIE verankert: Fold k testet fuer jede Serie deren
    eigenes Zeitfenster
        (last_s - (n_folds-k+1)*step,  last_s - (n_folds-k)*step]
    und trainiert auf allen Zielen <= Fensterstart. Auf gleichlangen Panels
    (Synthetik) ist das identisch mit globalen Kalender-Folds; auf Panelen
    mit ungleichen Serienenden (echte Daten) wird jede Serie ueber ihre
    volle Traegerbreite evaluiert statt nur nahe des globalen Panel-Endes.

    Parameters
    ----------
    df : Panel mit ``series``, ``date``, ``value`` (+ optionale exogene Spalten).
    horizons : Zu evaluierende Forecast-Horizonte.
    specs : Zu vergleichende Modelle (Default: LGBM + beide Baselines).
    n_folds : Anzahl der Folds (siehe oben).
    step_months : Fold-Breite in Monaten (Default: max(horizons)).
    origin : "rolling" (Default) - Baselines prognostizieren jede Zeile aus
             ihrem eigenen cutoff (= target_date - h), wie die LGBM-Zeilen.
             "fixed" - LGBM testet nur Zeilen mit cutoff == fold_end,
             Baselines nutzen nur Historie <= fold_end (M4-Protokoll).
    collect_importance : Gain-Importance der LGBM-Modelle mitsammeln.
    verbose : Fortschritt ausgeben.
    """
    if origin not in ("rolling", "fixed"):
        raise ValueError(f"origin muss 'rolling' oder 'fixed' sein, got {origin!r}")
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
    last_by_series = df.groupby("series")["date"].max()

    def _fold_ends(k: int) -> pd.Series:
        """Fold-Ende (Fensterstart) je Serie fuer Fold k."""
        return last_by_series - pd.DateOffset(months=(n_folds - k + 1) * step)

    pred_frames: list[pd.DataFrame] = []
    imp_rows: list[pd.DataFrame] = []

    for i in range(1, n_folds + 1):
        fe = _fold_ends(i)
        hi = fe + pd.DateOffset(months=step)

        # Per-Serie-Fenster als Basisframe (series, target_date).
        window_parts = []
        for s in last_by_series.index:
            win = grid[(grid > fe[s]) & (grid <= hi[s])]
            if len(win):
                window_parts.append(pd.DataFrame({"series": s, "target_date": win}))
        target_window_frame = (
            pd.concat(window_parts, ignore_index=True)
            if window_parts
            else pd.DataFrame(columns=["series", "target_date"])
        )

        for spec in specs:
            if spec.kind == "lgbm":
                sup = sup_cache[_config_key(spec.config or FeatureConfig())]
                sup_fe = sup["series"].map(fe)
                sup_hi = sup["series"].map(hi)
                train = sup[sup["target_date"] <= sup_fe]
                if origin == "fixed":
                    # Blockprognose vom eigenen fold_end je Serie.
                    test = sup[sup["date"] == sup_fe]
                else:
                    test = sup[
                        (sup["target_date"] > sup_fe) & (sup["target_date"] <= sup_hi)
                    ]
                if train.empty or test.empty:
                    continue
                model = DirectLGBM(
                    horizons=horizons, categorical=spec.categorical, params=spec.params or {}
                )
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
                base = _baseline_predictions(
                    df,
                    target_window_frame,
                    spec.kind,
                    horizons=horizons,
                    origin=origin,
                    fold_ends=fe,
                )
                out = base[["series", "cutoff", "target_date", "horizon", "y_ref", "y", "pred"]]

            out = out.rename(columns={"date": "cutoff"}).assign(model=spec.name, fold=i)
            cols = ["model", "fold", "series", "cutoff", "target_date", "horizon",
                    "y_ref", "y", "pred"]
            out = out[[c for c in cols if c in out.columns]]
            pred_frames.append(out)

            if verbose:
                print(
                    f"fold {i}/{n_folds} end=[{fe.min().date()} .. {fe.max().date()}] "
                    f"spec={spec.name} rows={len(out)}"
                )

    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    predictions, metrics_by_horizon, fold_metrics = evaluate_predictions(predictions)
    importance = pd.concat(imp_rows, ignore_index=True) if imp_rows else None
    return BacktestResult(predictions, metrics_by_horizon, fold_metrics, importance)


if __name__ == "__main__":
    from ..data.load import load_dataset

    panel = load_dataset("synthetic", n_series=12, n_periods=132)
    res = expanding_backtest(panel, horizons=(1, 6, 12, 18), n_folds=3, verbose=True)
    print(res.metrics_by_horizon.pivot(index="horizon", columns="model", values="mae"))
