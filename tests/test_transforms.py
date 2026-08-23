"""Round-Trips fuer TargetTransform und M4-Startdatum-Parsing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lgbm_panel.data.load import _parse_start_months
from lgbm_panel.features import TargetTransform, duan_smear_factor


@pytest.mark.parametrize("kind", ["identity", "log", "log1p", "seasdiff", "log_seasdiff"])
def test_roundtrip(kind):
    rng = np.random.default_rng(3)
    tf = TargetTransform(kind)
    if not tf.is_diff:
        y = pd.Series(rng.uniform(0.5, 500, 300))
        back = tf.inverse(tf.transform_values(y).to_numpy())
        assert np.allclose(back, y.to_numpy(), atol=1e-9)
    else:
        t = np.arange(60)
        s = pd.Series(50 + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 60))
        panel = pd.DataFrame(
            {"series": "A", "date": pd.date_range("2020-01-01", periods=60, freq="MS"), "value": s}
        )
        tp = tf.transform_panel(panel)
        z = tp["value"].to_numpy()[12:]
        back = tf.inverse(z, s.to_numpy()[:-12])
        assert np.allclose(back, s.to_numpy()[12:], atol=1e-9)


def test_transform_panel_per_series_diff():
    # Differenz je Serie, nicht ueber Seriengrenzen hinweg.
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    panel = pd.DataFrame(
        {
            "series": ["A"] * 24 + ["B"] * 24,
            "date": list(dates) * 2,
            "value": np.arange(24).tolist() + (100 + np.arange(24)).tolist(),
        }
    )
    tp = TargetTransform("seasdiff").transform_panel(panel)
    a = tp[tp["series"] == "A"].reset_index(drop=True)
    b = tp[tp["series"] == "B"].reset_index(drop=True)
    assert np.isnan(a["value"].iloc[:12]).all()
    assert a["value"].iloc[12:].tolist() == ([np.nan] * 0 + [12.0] * 12)
    assert b["value"].iloc[13] == pytest.approx(112 - 100)


def test_seasdiff_requires_season():
    with pytest.raises(ValueError):
        TargetTransform("seasdiff", season=0)


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        TargetTransform("bogus")


def test_pointwise_guard_for_diff_kinds():
    with pytest.raises(ValueError, match="transform_panel"):
        TargetTransform("log_seasdiff").transform_values(pd.Series([1.0]))


def test_smearing_factor_matches_analytic_moments():
    rng = np.random.default_rng(5)
    resid = rng.normal(0, 0.25, 200_000)
    factor = duan_smear_factor(resid)
    assert factor == pytest.approx(float(np.exp(0.25**2 / 2)), rel=5e-3)
    assert duan_smear_factor(np.array([])) == 1.0
    assert duan_smear_factor(np.array([np.nan, np.inf])) == 1.0


def test_log_inverse_applies_smear():
    tf = TargetTransform("log", smear=1.05)
    assert np.allclose(tf.inverse(np.array([0.0])), [1.05])


# --------------------------------------------------------------------------- #
# M4 Startdatum-Parsing
# --------------------------------------------------------------------------- #
def test_parse_two_digit_years_pivot():
    p = _parse_start_months(pd.Series({"M1": "01-06-76 12:00", "M2": "01-01-85 12:00"}))
    assert list(p) == list(pd.PeriodIndex(["1976-06", "1985-01"], freq="M").astype("int64"))


def test_parse_iso_dates():
    p = _parse_start_months(pd.Series({"M1": "1879-03-01 12:00:00"}))
    assert int(p.iloc[0]) == int(pd.Period("1879-03", freq="M").ordinal)


def test_century_correction_uses_series_length():
    # "31-01-90" mit 2794 Monatsbeobachtungen kann nicht 1990 sein:
    # 1990 + 233 Jahre laege weit hinter dem Datenhorizont -> 1790.
    raw = pd.Series({"LONG": "31-01-90 12:00"})
    n_obs = pd.Series({"LONG": 2794})
    p = _parse_start_months(raw, n_obs)
    start = pd.Period(ordinal=int(p.iloc[0]), freq="M")
    end = start + 2794 - 1
    assert end.year <= 2030
    assert start.year < 1900
    # Kurze Serie bleibt beim %y-Pivot: "01-01-13" + 90 Monate -> 2013..2020.
    raw2 = pd.Series({"S": "01-01-13 12:00"})
    p2 = _parse_start_months(raw2, pd.Series({"S": 90}))
    assert pd.Period(ordinal=int(p2.iloc[0]), freq="M") == pd.Period("2013-01", freq="M")


def test_unparsable_date_raises():
    with pytest.raises(ValueError, match="Unparsbare"):
        _parse_start_months(pd.Series({"X": "not-a-date"}))
