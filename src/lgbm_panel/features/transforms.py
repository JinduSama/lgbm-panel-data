"""
Ziel-Transformationen für Panel-Forecasting.

Ein ``TargetTransform`` kapselt die Vorwaerts- (Levels -> Modellraum) und die
Ruecktransformation (Modellraum -> Levels) inklusive optionaler
Retransformations-Bias-Korrektur nach Duan (1983) fuer Log-Zielformen:
``exp(E[log y])`` systematisch unterschaetzt ``E[y]``; der Smearing-Faktor
``mean(exp(resid))`` aus den Trainingsresiduen korrigiert das.

Formen:
    identity      : y
    log           : log(y)
    log1p         : log1p(y)               (auch fuer Werte nahe 0)
    seasdiff      : y_t - y_{t-season};    Rueck: Anker + z
    log_seasdiff  : log(y_t) - log(y_{t-season}); Rueck: Anker * exp(z)

Bei den Differenzformen ist der Anker der beobachtete Wert ``season``
Perioden vor dem Ziel - die Ruecktransformation ist also genau dann
leakage-frei, wenn ``h <= season``.

Verwendung:
    from lgbm_panel.features import TargetTransform
    tf = TargetTransform("log_seasdiff")
    train_df = tf.transform_panel(panel)
    # ... LGBM auf train_df ...
    levels = tf.inverse(pred_z, anchor=y[target - tf.season])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KINDS = ("identity", "log", "log1p", "seasdiff", "log_seasdiff")


def duan_smear_factor(residuals: np.ndarray | pd.Series) -> float:
    """
    Duan-Smearing-Faktor fuer Log-Zielformen: Mittelwert von exp(resid).

    Residuen im Transformationsraum (z.B. log), NaNs werden ignoriert.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 1.0
    return float(np.mean(np.exp(r)))


@dataclass
class TargetTransform:
    """Vorwaerts-/Ruecktransformation der Zielgroesse eines Panels."""

    kind: str = "identity"
    season: int = 12
    # Multiplikativer Smearing-Faktor fuer Log-Formen (1.0 = aus).
    smear: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind muss einer von {KINDS} sein, got {self.kind!r}")
        if self.kind in ("seasdiff", "log_seasdiff") and self.season < 1:
            raise ValueError("season >= 1 fuer Differenzformen")

    @property
    def is_diff(self) -> bool:
        return self.kind in ("seasdiff", "log_seasdiff")

    def transform_values(self, values: pd.Series) -> pd.Series:
        """
        Punktweise Vorwaertstransformation (nur Nicht-Differenzformen).

        Fuer seasdiff/log_seasdiff bitte ``transform_panel`` nutzen.
        """
        if self.is_diff:
            raise ValueError(f"{self.kind!r} benoetigt transform_panel (Seriendiff)")
        v = values.to_numpy(dtype=float)
        if self.kind == "log":
            v = np.log(np.clip(v, a_min=1e-12, a_max=None))
        elif self.kind == "log1p":
            v = np.log1p(np.clip(v, a_min=-1 + 1e-12, a_max=None))
        return pd.Series(v, index=values.index)

    def transform_panel(self, df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
        """
        Wendet die Transformation auf ein Long-Panel an (Spalten series/date/value).

        Bei Differenzformen wird je Serie um ``season`` differenziert; die
        ersten ``season`` Zeilen jeder Serie erhalten NaN (build_supervised
        verwirft sie via dropna auf dem Ziel automatisch).
        """
        out = df.copy()
        if not self.is_diff:
            out[value_col] = self.transform_values(out[value_col])
            return out
        g = out.sort_values(["series", "date"]).groupby("series", sort=False)[value_col]
        if self.kind == "seasdiff":
            out[value_col] = g.diff(self.season)
        else:  # log_seasdiff: erst punkweise loggen (guard umgehen), dann diff
            v = out[value_col].to_numpy(dtype=float)
            logged = pd.Series(np.log(np.clip(v, a_min=1e-12, a_max=None)), index=out.index)
            out[value_col] = logged.groupby(out["series"], sort=False).diff(self.season)
        return out

    def inverse(
        self,
        z: np.ndarray | pd.Series,
        anchor: np.ndarray | pd.Series | float = 0.0,
    ) -> np.ndarray:
        """
        Modellraum -> Levels.

        Parameters
        ----------
        z : Vorhersage (oder Beobachtung) im Transformationsraum.
        anchor : Beobachteter Level-Anker. Fuer Differenzformen der Wert
            ``season`` Perioden vor dem Ziel; sonst ungenutzt.

        Returns
        -------
        np.ndarray mit Level-Vorhersagen (inkl. Smearing bei Log-Formen,
        falls ``smear`` != 1.0 gesetzt wurde).
        """
        z = np.asarray(z, dtype=float)
        if self.kind == "identity":
            return z
        if self.kind == "log":
            return np.exp(z) * self.smear
        if self.kind == "log1p":
            return np.expm1(z) * self.smear
        a = np.asarray(anchor, dtype=float)
        if self.kind == "seasdiff":
            return a + z
        # log_seasdiff
        return np.exp(np.log(np.clip(a, a_min=1e-12, a_max=None)) + z) * self.smear


if __name__ == "__main__":
    from lgbm_panel.data import load_dataset

    panel = load_dataset("synthetic", n_series=3, n_periods=48)
    tf = TargetTransform("log_seasdiff")
    print(tf.transform_panel(panel).head(15))
