"""Edge cases fuer Metriken: MAE/RMSE/sMAPE/Directional Accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from lgbm_panel.metrics import directional_accuracy, mae, rmse, smape


def test_mae_rmse_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert mae(y, p) == pytest.approx(2.0 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt((0 + 0 + 4) / 3))


def test_smape_zero_denominator_guard():
    # y=pred=0 -> denom wuerde zu 0 kollabieren; muss nicht inf/nan werden.
    v = smape(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert np.isfinite(v)


def test_dir_acc_perfect_and_inverse():
    y_ref = np.array([10.0, 10.0])
    y_true = np.array([11.0, 9.0])
    assert directional_accuracy(y_true, y_true + 1, y_ref=y_ref) == pytest.approx(1.0)
    # Vorhersage bewegt sich entgegengesetzt zur Wahrheit -> 0.0.
    y_wrong = np.where(y_true > y_ref, y_ref - 1.0, y_ref + 1.0)
    assert directional_accuracy(y_true, y_wrong, y_ref=y_ref) == pytest.approx(0.0)


def test_dir_acc_excludes_zero_true_movement():
    # Bewegung 0 in Wahrheit -> Zeile zaehlt nicht (alter Bestand).
    y_ref = np.array([10.0, 10.0])
    y_true = np.array([10.0, 11.0])  # erste Zeile: keine wahre Bewegung
    y_pred = np.array([99.0, 12.0])  # erste Zeile voellig falsch, aber egal
    assert directional_accuracy(y_true, y_pred, y_ref=y_ref) == pytest.approx(1.0)


def test_dir_acc_excludes_zero_predicted_movement():
    # P0.2: snaive @ h=12 hat pred == y_ref -> d_pred = 0 -> Zeile muss
    # ausgeschlossen sein (vorher wurde sie als FALSCH gezaehlt -> dir_acc 0).
    y_ref = np.array([10.0, 10.0])
    y_true = np.array([11.0, 9.0])  # Richtungen +1/-1
    y_pred = np.array([10.0, 10.0])  # Vorhersage: keine Richtung
    assert np.isnan(directional_accuracy(y_true, y_pred, y_ref=y_ref))


def test_dir_acc_snaive_h12_scenario_is_nan():
    # Exakt die E5-Situation: alle Richtungen undefiniert -> NaN, nicht 0.
    rng = np.random.default_rng(0)
    y_ref = rng.uniform(1, 100, 50)
    acc = directional_accuracy(y_ref + rng.normal(0, 1, 50), y_ref.copy(), y_ref=y_ref)
    assert np.isnan(acc)


def test_dir_acc_nan_reference_excluded():
    y_ref = np.array([np.nan, 5.0])
    y_true = np.array([6.0, 6.0])
    y_pred = np.array([7.0, 7.0])
    assert directional_accuracy(y_true, y_pred, y_ref=y_ref) == pytest.approx(1.0)


def test_dir_acc_scalar_reference():
    rng = np.random.default_rng(1)
    y_true = np.arange(1.0, 11.0)
    y_pred = y_true + rng.normal(0, 0.1, 10)
    acc = directional_accuracy(y_true, y_pred, y_ref=0.0)
    assert acc == pytest.approx(1.0)


def test_dir_acc_empty_returns_nan():
    assert np.isnan(directional_accuracy([], [], y_ref=[]))
