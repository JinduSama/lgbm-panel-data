"""Leakage- und Form-Tests fuer build_supervised."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lgbm_panel.features import FeatureConfig, build_supervised, feature_columns


@pytest.fixture()
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=48, freq="MS")
    frames = []
    for s in range(3):
        frames.append(
            pd.DataFrame(
                {
                    "series": f"S{s}",
                    "date": dates,
                    "value": 100 + rng.normal(0, 5, 48).cumsum(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_lags_are_strictly_past(panel):
    # Leakage-Regression: lag_k der Zeile mit Cutoff t muss exakt value[t-k]
    # sein; y_h muss exakt value[t+h] sein. Ein Off-by-One waere Leakage.
    data = build_supervised(panel, horizons=(1, 6, 12))
    sample = data.sample(100, random_state=0)
    lookup = panel.set_index(["series", "date"])["value"]
    for _, row in sample.iterrows():
        for k in (1, 2, 3, 6, 12):
            assert row[f"lag_{k}"] == pytest.approx(
                lookup.loc[(row["series"], row["date"] - pd.DateOffset(months=k))]
            )
        h = int(row["horizon"])
        assert row["y"] == pytest.approx(
            lookup.loc[(row["series"], row["date"] + pd.DateOffset(months=h))]
        )


def test_rolling_windows_end_at_cutoff(panel):
    data = build_supervised(panel, horizons=(1,))
    g = panel.set_index(["series", "date"])["value"]
    row = data.iloc[30]
    window = [g.loc[(row["series"], row["date"] - pd.DateOffset(months=k))] for k in range(3)]
    assert row["roll3_mean"] == pytest.approx(np.mean(window))


def test_no_row_uses_target_as_feature(panel):
    # Feature-Zeilen duerfen value[t] selbst nicht enthalten (nur lags/rollings).
    cfg = FeatureConfig(exog_cols=[])
    data = build_supervised(panel, horizons=(1,), config=cfg)
    lookup = panel.set_index(["series", "date"])["value"]
    own = np.array([lookup.loc[(r["series"], r["date"])] for _, r in data.iterrows()])
    for c in ("lag_1", "roll3_mean"):
        assert not np.allclose(data[c].to_numpy(), own)

def test_diff_nans_retained_but_rows_with_nan_y_dropped(panel):
    # Mit reduziertem Lag-Bedarf ueberleben die ersten Monate je Serie;
    # dort ist diff_12 absichtlich NaN (LightGBM-Handling), y aber beobachtet.
    cfg = FeatureConfig(lags=(1,), rolling_windows=(), rolling_stats=(), diff_lags=(12,))
    data = build_supervised(panel, horizons=(1, 12), config=cfg)
    first = data.groupby("series")["date"].transform("min")
    early = data[data["date"] < first + pd.DateOffset(months=11)]
    assert not early.empty
    assert early["diff_12"].isna().all()
    assert early["y"].notna().all()


def test_horizon_grid_trails_off(panel):
    # Laengere Horizonte verlieren die Ziele hinter dem Panel-Ende.
    data = build_supervised(panel, horizons=(1, 6, 12))
    counts = data.groupby("horizon").size()
    assert counts[1] >= counts[6] >= counts[12] > 0




def test_feature_columns_matches_frame(panel):
    cfg = FeatureConfig()
    data = build_supervised(panel, horizons=(1,), config=cfg)
    cols = feature_columns(data, cfg)
    missing = [c for c in cols if c not in data.columns]
    assert not missing
