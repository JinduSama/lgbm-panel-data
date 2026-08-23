"""Per-Serie Zeit-Split fuer Early Stopping: Vertrag des Helfers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lgbm_panel.features import build_supervised
from lgbm_panel.strategies.direct_forecast import DirectLGBM, per_series_time_split


@pytest.fixture()
def ragged_supervised():
    # Zwei Serien mit sehr unterschiedlichen Enden; echte Feature-Spalten.
    rng = np.random.default_rng(2)
    dates_a = pd.date_range("2018-01-01", periods=80, freq="MS")
    dates_b = pd.date_range("2020-01-01", periods=40, freq="MS")
    frames = []
    for name, dates in (("A", dates_a), ("B", dates_b)):
        frames.append(
            pd.DataFrame(
                {
                    "series": name,
                    "date": dates,
                    "value": 100 + rng.normal(0, 5, len(dates)).cumsum(),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    return build_supervised(panel, horizons=(1,))


def test_split_keeps_both_series_on_both_sides(ragged_supervised):
    # Der alte gepoolte Datum-Quantil schob die kurze, spaet startende Serie B
    # KOMPLETT in die Validierung. Der per-Serie Split darf das nicht.
    mask = per_series_time_split(ragged_supervised, valid_fraction=0.25)
    df = ragged_supervised.assign(_fit=mask)
    for s, grp in df.groupby("series"):
        assert grp["_fit"].any(), f"Serie {s} fehlt im Fit"
        assert (~grp["_fit"]).any(), f"Serie {s} fehlt in der Validierung"


def test_split_is_temporal_tail_per_series(ragged_supervised):
    mask = per_series_time_split(ragged_supervised, valid_fraction=0.25)
    df = ragged_supervised.assign(_fit=mask)
    for s, grp in df.groupby("series"):
        fit_max = grp.loc[grp["_fit"], "date"].max()
        val_min = grp.loc[~grp["_fit"], "date"].min()
        assert val_min > fit_max, f"Serie {s}: Validierung liegt nicht zeitlich nach dem Fit"
        n_val = int((~grp["_fit"]).sum())
        assert n_val == pytest.approx(0.25 * len(grp), abs=1)


def test_fit_with_valid_fraction_trains(ragged_supervised):
    # Smoke: valid_fraction-Pfad end-to-end (per-Serie Split im fit).
    model = DirectLGBM(horizons=(1,), categorical=("series",))
    model.fit(ragged_supervised, valid_fraction=0.25, num_boost_round=20)
    preds = model.predict(ragged_supervised)
    assert preds["pred"].notna().all()
