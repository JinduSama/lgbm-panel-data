"""
Feature-Engineering für Panel-Forecasting mit LightGBM (vektorisiert).

Direkte Multi-Horizon-Formulierung:
    Fuer jeden Cutoff-Zeitpunkt t und Horizont h (1..H):
        Features  = Lags + Rolling-Stats + Zeitfeatures (alle nur Vergangenheit)
        Ziel      = value[t + h]
    -> Eine Zeile pro (series, cutoff, horizon). Kein Leakage.

Verwendung:
    from lgbm_panel.features import FeatureConfig, build_supervised
    data = build_supervised(df, horizons=(1, 6, 12, 18))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureConfig:
    """Konfiguration für das Feature-Engineering."""

    lags: Sequence[int] = (1, 2, 3, 6, 12, 13, 18, 24)
    rolling_windows: Sequence[int] = (3, 6, 12)
    rolling_stats: tuple[str, ...] = ("mean", "std", "min", "max")
    time_features: tuple[str, ...] = ("month", "quarter", "year")
    diff_lags: Sequence[int] = (1, 12)
    use_cross_sectional: bool = False
    exog_lags: tuple[int, ...] = (0, 1)
    exog_cols: Sequence[str] | None = None  # None = automatisch erkennen
    # Szenario-Exogena: Werte bei (target_date - j). NUR nutzen, wenn diese
    # Zukuempfe zum Forecast-Zeitpunkt bekannt sind (z.B. geplante Budgets).
    exog_scenario_lags: Sequence[int] = ()


def _add_time_features(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    dt = df["date"].dt
    for name in cols:
        df[name] = getattr(dt, name)
    return df


def build_supervised(
    df: pd.DataFrame,
    horizons: Sequence[int],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Baut die supervised Tabelle für direktes Multi-Horizon-Forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Panel-Daten mit Spalten ``series``, ``date``, ``value``.
    horizons : Sequence[int]
        Forecast-Horizonte (z.B. ``(1, 6, 12, 18)``).
    config : FeatureConfig, optional

    Returns
    -------
    pd.DataFrame
        Eine Zeile pro (series, cutoff, horizon) mit Feature-Spalten und
        der Ziel-Spalte ``y`` (= value zum Ziel-Datum).
    """
    cfg = config or FeatureConfig()
    df = df.sort_values(["series", "date"]).reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    g = df.groupby("series", sort=False)["value"]

    # --- Lags (nur Vergangenheit relativ zum Cutoff) -------------------
    max_lag = max(cfg.lags)
    for lag in cfg.lags:
        df[f"lag_{lag}"] = g.shift(lag)

    # --- Rolling-Stats (Fenster endet beim Cutoff) ---------------------
    for w in cfg.rolling_windows:
        for stat in cfg.rolling_stats:
            df[f"roll{w}_{stat}"] = g.transform(
                lambda s, w=w, stat=stat: getattr(s.rolling(w, min_periods=w), stat)()
            )

    # --- Differenzen (Saisondifferenz -12) -----------------------------
    for d in cfg.diff_lags:
        df[f"diff_{d}"] = g.diff(d)
    # --- Exogene Treiber (am Cutoff beobachtet -> leakage-frei) --------
    generated = {"series", "date", "value"}
    prefix = ("lag_", "roll", "diff_", "y_h", "target_date", "xs_")
    if cfg.exog_cols is not None:
        exog_cols = [
            c
            for c in cfg.exog_cols
            if c in df.columns and c not in generated and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        exog_cols = [
            c
            for c in df.columns
            if c not in generated
            and not c.startswith(prefix)
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    exog_feat_cols: list[str] = []
    for col in exog_cols:
        gc = df.groupby("series", sort=False)[col]
        for k in cfg.exog_lags:
            name = col if k == 0 else f"{col}_lag{k}"
            df[name] = gc.shift(k)
            exog_feat_cols.append(name)

    # --- Cross-sectionale Aggregate (Panel-Mittel je Datum, t ist bekannt)
    if cfg.use_cross_sectional:
        xs = df.groupby("date")["value"]
        df["xs_mean"] = xs.transform("mean")
        df["xs_std"] = xs.transform("std")

    # --- Zeitfeatures des Cutoffs --------------------------------------
    df = _add_time_features(df, cfg.time_features)

    # --- Ziel-Spalten: value[t + h] fuer jedes h -----------------------
    for h in horizons:
        df[f"y_h{h}"] = g.shift(-h)
        df[f"target_date_h{h}"] = df.groupby("series", sort=False)["date"].shift(-h)

    # --- Zeilen mit unvollstaendiger Historie verwerfen ----------------
    needed = [f"lag_{lag}" for lag in cfg.lags if lag <= max_lag]
    df = df.dropna(subset=needed).reset_index(drop=True)

    # --- Long-Format: eine Zeile pro (cutoff, horizon) -----------------
    id_cols = ["series", "date"] + list(cfg.time_features)
    feat_cols = (
        [f"lag_{lag}" for lag in cfg.lags]
        + [f"roll{w}_{stat}" for w in cfg.rolling_windows for stat in cfg.rolling_stats]
        + [f"diff_{d}" for d in cfg.diff_lags]
        + exog_feat_cols
        + (["xs_mean", "xs_std"] if cfg.use_cross_sectional else [])
    )

    frames = []
    for h in horizons:
        part = df[id_cols + feat_cols].copy()
        part["horizon"] = h
        part["y"] = df[f"y_h{h}"]
        part["target_date"] = df[f"target_date_h{h}"]
        frames.append(part)

    out = pd.concat(frames, ignore_index=True)
    # --- Szenario-Exogena: x bei (target_date - j) je Zeile -------------
    if cfg.exog_scenario_lags and exog_cols:
        lookup_x = df.set_index(["series", "date"])[exog_cols]
        for col in exog_cols:
            for j in cfg.exog_scenario_lags:
                key_date = out["target_date"] - pd.DateOffset(months=int(j))
                out[f"{col}_at_tminus{j}"] = (
                    lookup_x[col]
                    .reindex(pd.MultiIndex.from_arrays([out["series"], key_date]))
                    .to_numpy()
                )
    return out.dropna(subset=["y"]).reset_index(drop=True)


def feature_columns(data: pd.DataFrame, cfg: FeatureConfig | None = None) -> list[str]:
    """Gibt die Liste der Feature-Spalten (inkl. Exogena/Cross-Sectional) zurueck."""
    cfg = cfg or FeatureConfig()
    base = (
        [f"lag_{lag}" for lag in cfg.lags]
        + [f"roll{w}_{stat}" for w in cfg.rolling_windows for stat in cfg.rolling_stats]
        + [f"diff_{d}" for d in cfg.diff_lags]
        + list(cfg.time_features)
    )
    if cfg.use_cross_sectional:
        base += ["xs_mean", "xs_std"]
    known = set(base) | {"series", "date", "horizon", "y", "target_date", "y_ref"}
    known |= {c for c in data.columns if c.startswith(("y_h", "target_date_h"))}
    numeric = [c for c in data.columns if c not in known and pd.api.types.is_numeric_dtype(data[c])]
    if cfg.exog_cols is None:
        # Auto-Modus: alle numerischen Extras sind Exogena.
        return base + sorted(numeric)
    # Explizite Auswahl: nur die konfigurierten Exogena und deren Ableitungen.
    explicit: list[str] = []
    for col in cfg.exog_cols:
        for name in (
            [col]
            + [f"{col}_lag{k}" for k in cfg.exog_lags if k != 0]
            + [f"{col}_at_tminus{j}" for j in cfg.exog_scenario_lags]
        ):
            if name in data.columns:
                explicit.append(name)
    return base + sorted(set(explicit))


if __name__ == "__main__":
    from lgbm_panel.data import load_dataset

    df = load_dataset("synthetic", n_series=3, n_periods=48)
    data = build_supervised(df, horizons=(1, 6, 12))
    print(data.shape)
    print(data.head())
