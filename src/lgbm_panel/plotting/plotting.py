"""
Visualisierungen für LGBM Panel-Forecasting.

Plotet Vorhersage vs. Ist, Feature-Bedeutung und Backtest-Metriken über die Zeit.

Verwendung:
    from lgbm_panel.plotting import plot_forecast, plot_feature_importance, plot_backtest_metrics
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    actual: pd.Series,
    predicted: pd.Series,
    horizon: int = 18,
    group: str | None = None,
    title: str = "Vorhersage vs. Ist",
) -> None:
    """
    Plotet Vorhersage vs. Ist für eine Zeitreihe.

    Parameters
    ----------
    actual : pd.Series
        Tatsächliche Werte.
    predicted : pd.Series
        Vorhersagen (indexiert nach ``date``).
    horizon : int
        Prognosehorizont.
    group : str, optional
        Gruppen-ID (optional im Titel).
    title : str
        Diagrammtitel.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual.index, actual.values, label="Ist", linewidth=2)
    for h in range(1, horizon + 1):
        if h in predicted.index:
            ax.plot(
                predicted.index,
                predicted[h].values,
                label=f"Horizont {h}",
                alpha=0.7,
                linewidth=1,
            )
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig)


def plot_feature_importance(importances: pd.DataFrame, title: str = "Feature-Bedeutung") -> None:
    """
    Plotet die Feature-Bedeutung.

    Parameters
    ----------
    importances : pd.DataFrame
        DataFrame mit Spalten ``feature`` und ``importance``.
    title : str
        Diagrammtitel.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.sort_values("importance", ascending=True).plot(
        kind="barh", ax=ax, color="steelblue"
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig)


def plot_backtest_metrics(
    metrics: pd.DataFrame, title: str = "Backtest-Metriken über die Zeit"
) -> None:
    """
    Plotet Backtest-Metriken über die Zeit.

    Parameters
    ----------
    metrics : pd.DataFrame
        DataFrame mit Metriken pro Fold (z. B. ``fold``, ``mae``, ``rmse``).
    title : str
        Diagrammtitel.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics.plot(kind="line", ax=ax, marker="o")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig)


def _save_or_show(fig) -> None:
    """Speichert das Diagramm oder zeigt es an."""
    out_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "forecast.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagramm gespeichert: {out_dir / 'forecast.png'}")
