"""Protokoll-Tests fuer expanding_backtest: rolling vs. fixed Origin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lgbm_panel.data import make_panel
from lgbm_panel.experiments import ModelSpec, expanding_backtest

HORIZONS = (1, 6, 12)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel(
        n_series=6,
        n_periods=84,
        horizon=max(HORIZONS),
        seed=11,
        trend_growth=(0.008, 0.02),
        seasonal_strength=(20.0, 40.0),
        noise_scale=(2.0, 5.0),
    )


def _baseline_frame(res, model: str) -> pd.DataFrame:
    out = res.predictions[res.predictions["model"] == model]
    assert not out.empty
    return out


def test_rolling_naive_predicts_value_at_own_cutoff(panel):
    res = expanding_backtest(panel, horizons=HORIZONS, n_folds=2, origin="rolling")
    naive = _baseline_frame(res, "naive")
    lookup = panel.set_index(["series", "date"])["value"]
    sample = naive.sample(150, random_state=0)
    expected = [
        lookup.loc[(r["series"], r["cutoff"])] for _, r in sample.iterrows()
    ]
    assert np.allclose(sample["pred"].to_numpy(), expected)


def test_rolling_snaive_uses_seasonal_lag_within_cutoff(panel):
    res = expanding_backtest(panel, horizons=HORIZONS + (18,), n_folds=2, origin="rolling")
    snaive = _baseline_frame(res, "seasonal_naive")
    lookup = panel.set_index(["series", "date"])["value"]
    for _, r in snaive.sample(150, random_state=1).iterrows():
        h = int(r["horizon"])
        k = -(-h // 12)  # letzte Periode mit gleichem Kalendermonat <= cutoff
        src = r["target_date"] - pd.DateOffset(months=12 * k)
        assert r["pred"] == pytest.approx(lookup.loc[(r["series"], src)])
        assert src <= r["cutoff"]  # kein Leakage: Quelle liegt am/am vor cutoff


def test_support_is_reported_and_common(panel):
    res = expanding_backtest(panel, horizons=HORIZONS, n_folds=2, origin="rolling")
    m = res.metrics_by_horizon
    assert {"n", "mae", "rmse", "smape", "dir_acc"} <= set(m.columns)
    # Gemeinsame Unterlage: identisches n je Modell und Horizont.
    piv = m.pivot(index="horizon", columns="model", values="n")
    assert (piv.nunique(axis=1) == 1).all()
    # Tatsaechlich exakt gleiche Schluesselmenge je Modell:
    keys = ["fold", "horizon", "series", "target_date"]
    sets = {m_: set(map(tuple, g[keys].itertuples(index=False, name=None)))
            for m_, g in res.predictions.groupby("model")}
    first = next(iter(sets.values()))
    for other in sets.values():
        assert other == first


def test_fixed_origin_restricts_lgbm_to_fold_end(panel):
    res = expanding_backtest(panel, horizons=HORIZONS, n_folds=2, origin="fixed")
    lgbm_rows = _baseline_frame(res, "lgbm")
    grid = pd.DatetimeIndex(sorted(panel["date"].unique()))
    fold_ends = [grid[-1] - pd.DateOffset(months=12), grid[-1] - pd.DateOffset(months=24)]
    # cutoff == fold_end je Fold
    for _, r in lgbm_rows.iterrows():
        assert r["cutoff"] in fold_ends, f"cutoff {r['cutoff']} ist kein Fold-Ende"
    # Ziele liegen strikt im Testfenster hinter dem Fold-Ende
    assert (lgbm_rows["target_date"] > lgbm_rows["cutoff"]).all()


def test_frozen_baseline_in_fixed_mode_uses_only_history(panel):
    res = expanding_backtest(
        panel,
        horizons=(1,),
        n_folds=2,
        origin="fixed",
        specs=[ModelSpec(name="naive", kind="naive")],
    )
    naive = _baseline_frame(res, "naive")
    lookup = panel.set_index(["series", "date"])["value"]
    # Fixed-Naive prognostiziert den letzten Wert <= fold_end fuer beide Folds;
    # der Wert muss auf JEDEM Weg vor dem Ziel liegen und beobachtet sein.
    for _, r in naive.iterrows():
        assert r["pred"] == pytest.approx(lookup.loc[(r["series"], r["cutoff"])])
        assert r["cutoff"] < r["target_date"]


def test_invalid_origin_rejected(panel):
    with pytest.raises(ValueError, match="origin"):
        expanding_backtest(panel, horizons=HORIZONS, origin="bogus")


def test_lgbm_test_targets_never_leak_into_training_window(panel):
    # Trainingsziele enden am fold_end; Testziele liegen danach (beide Modi).
    for origin in ("rolling", "fixed"):
        res = expanding_backtest(panel, horizons=HORIZONS, n_folds=2, origin=origin)
        grid = pd.DatetimeIndex(sorted(panel["date"].unique()))
        step = max(HORIZONS)
        for i, fe in enumerate(
            [grid[-1] - pd.DateOffset(months=12 * k) for k in (2, 1)], start=1
        ):
            rows = res.predictions[
                (res.predictions["fold"] == i) & (res.predictions["model"] == "lgbm")
            ]
            if origin == "rolling":
                assert (rows["target_date"] > fe).all()
                assert (rows["target_date"] <= fe + pd.DateOffset(months=step)).all()
            else:
                assert (rows["cutoff"] == fe).all()
