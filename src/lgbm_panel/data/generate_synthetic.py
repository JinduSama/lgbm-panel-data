"""
Synthetische Panel-Daten mit kontrollierbarem Data Generating Process (DGP).

Jede Serie besteht aus:
    y[t] = level * exp(growth * t)            # exponentieller Trend
         + amp * sin(2*pi*t/12 + phase)       # jaehrliche Saisonalitaet
         + AR(1)-Rauschen
Der DGP ist bewusst "LGBM-freundlich" (starke Saisonalitaet, stabile Muster),
damit wir messen koennen, wie nah LGBM an der theoretisch erreichbaren
Genauigkeit kommt und welche Features was beitragen.

Verwendung:
    from lgbm_panel.data.generate_synthetic import make_panel
    df = make_panel(n_series=50, n_periods=96, horizon=18)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_panel(
    n_series: int = 50,
    n_periods: int = 96,
    horizon: int = 18,
    seed: int = 42,
    freq: str = "MS",
    start: str = "2015-01-01",
    seasonal_strength: tuple[float, float] = (0.0, 40.0),
    noise_scale: tuple[float, float] = (1.0, 12.0),
    ar_phi_range: tuple[float, float] = (-0.3, 0.6),
    trend_growth: tuple[float, float] = (-0.01, 0.02),
) -> pd.DataFrame:
    """
    Generiert ein synthetisches Monats-Panel.

    Returns
    -------
    pd.DataFrame
        Spalten: ``series``, ``date``, ``value`` (Long-Format, sortiert).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_periods, freq=freq)

    frames = []
    for s in range(n_series):
        t = np.arange(n_periods, dtype=float)

        level = rng.uniform(20, 500)
        growth = rng.uniform(*trend_growth)
        amp = rng.uniform(*seasonal_strength)
        phase = rng.uniform(0, 2 * np.pi)
        sigma = rng.uniform(*noise_scale)
        phi = rng.uniform(*ar_phi_range)

        trend_part = level * np.exp(growth * t)
        season_part = amp * np.sin(2 * np.pi * t / 12.0 + phase)

        eps = np.zeros(n_periods)
        shocks = rng.normal(0, sigma, n_periods)
        for i in range(1, n_periods):
            eps[i] = phi * eps[i - 1] + shocks[i]
        values = trend_part + season_part + eps

        frames.append(
            pd.DataFrame({"series": f"S{s:04d}", "date": dates, "value": values})
        )

    df = pd.concat(frames, ignore_index=True)
    return df


if __name__ == "__main__":
    data = make_panel()
    print(f"Zeilen: {data.shape[0]}, Serien: {data['series'].nunique()}")
    print(data.head())
