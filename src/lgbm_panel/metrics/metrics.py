"""
Metriken für LGBM Panel-Forecasting.

Berechnet MAE, RMSE, sMAPE und directional accuracy.

Verwendung:
    from lgbm_panel.metrics import mae, rmse, smape, directional_accuracy
    mae(y_true, y_pred)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Root Mean Squared Error."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def smape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    scale: float = 200.0,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true, y_pred : array-like
        Erwartete Werte.
    scale : float
        Skalierungsfaktor (Standard: 200 -> Ergebnis in Prozent).

    Returns
    -------
    float
        sMAPE-Wert.
    """
    denom = np.abs(np.asarray(y_true, dtype=float)) + np.abs(np.asarray(y_pred, dtype=float))
    denom[denom == 0] = 1e-8  # Verhindert Division durch Null.
    diff = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return float(scale * np.mean(diff / denom))


def directional_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_ref: pd.Series | np.ndarray | float | None = None,
) -> float:
    """
    Directional accuracy: Anteil der Vorhersagen mit korrekter Richtung.

    Richtung = Bewegung relativ zur Referenz (z.B. letzter beobachteter Wert
    zum Forecast-Origin). Standardmaessig wird der jeweils vorherige Wert von
    ``y_true`` als Referenz verwendet.

    Zeilen ohne definierte Richtung (Referenz NaN oder tatsaechliche Bewegung 0)
    werden aus der Berechnung ausgeschlossen.

    Returns
    -------
    float
        Anteil korrekt gerichteter Vorhersagen (0..1). NaN bei leerer Menge.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_ref is None:
        y_ref = np.concatenate(([np.nan], y_true[:-1]))
    elif np.isscalar(y_ref):
        y_ref = np.full(len(y_true), float(y_ref))
    else:
        y_ref = np.asarray(y_ref, dtype=float)

    d_true = np.sign(y_true - y_ref)
    d_pred = np.sign(y_pred - y_ref)
    valid = np.isfinite(y_ref) & (d_true != 0)
    if not valid.any():
        return float("nan")
    return float(np.mean(d_pred[valid] == d_true[valid]))
