"""
Forecast-Strategien für Panel-Daten.

- ``DirectLGBM``  : ein LightGBM-Modell pro Horizont (global ueber alle Serien).
- ``SeasonalNaive``: Baseline y[t+h] = y[t+h-12].
- ``Naive``        : Baseline y[t+h] = y[t].

Verwendung:
    from lgbm_panel.strategies import DirectLGBM, SeasonalNaive
    model = DirectLGBM(horizons=(1, 6, 12, 18))
    model.fit(train_data)
    preds = model.predict(test_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.build_features import FeatureConfig, feature_columns

DEFAULT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbosity": -1,
}


@dataclass
class DirectLGBM:
    """
    Direkte Multi-Horizon-Forecasts mit LightGBM.

    Trainiert ein globales Modell pro Horizont ueber alle Serien hinweg
    (Panel-Learning). ``series`` wird als kategoriales Feature uebergeben.
    """

    horizons: tuple[int, ...]
    params: dict = field(default_factory=dict)
    categorical: tuple[str, ...] = ("series",)
    models: dict[int, lgb.Booster] = field(default_factory=dict, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)
    _categories_: dict[str, list] = field(default_factory=dict, init=False)

    def fit(
        self,
        data: pd.DataFrame,
        config: FeatureConfig | None = None,
        num_boost_round: int = 500,
        valid_fraction: float = 0.0,
        **overrides,
    ) -> DirectLGBM:
        cfg = config or FeatureConfig()
        self.feature_names_ = feature_columns(data, cfg)
        cat = [c for c in self.categorical if c in data.columns]
        params = {**DEFAULT_PARAMS, **self.params, **overrides}

        # Kategorien einmalig aus den Trainingsdaten uebernehmen.
        self._categories_ = {c: pd.Categorical(data[c]).categories.tolist() for c in cat}

        for h in self.horizons:
            sub = data[data["horizon"] == h]
            if sub.empty:  # Horizont ohne trainierbare Zeilen (Historie zu kurz).
                continue
            X = sub[self.feature_names_ + cat].copy()
            for c in cat:
                X[c] = pd.Categorical(X[c], categories=self._categories_[c])
            y = sub["y"].values

            if valid_fraction > 0 and len(sub) > 200:
                # Zeitlich sortierter Split (letzte n% als Validierung).
                cutoff = sub["date"].quantile(1 - valid_fraction)
                mask = sub["date"] <= cutoff
                dtrain = lgb.Dataset(X[mask.values], label=y[mask.values])
                dvalid = lgb.Dataset(X[~mask.values], label=y[~mask.values])
                booster = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    valid_sets=[dvalid],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            else:
                dtrain = lgb.Dataset(X, label=y)
                booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)

            self.models[h] = booster
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vorhersagen fuer alle (series, cutoff, horizon)-Zeilen."""
        cat = [c for c in self.categorical if c in data.columns]
        keep = ["series", "date", "target_date", "horizon", "y"]
        keep += ["y_ref"] if "y_ref" in data.columns else []
        keep += [c for c in self.feature_names_ if c in data.columns]
        out = data[keep].copy()
        preds = np.full(len(data), np.nan)
        for h, booster in self.models.items():
            mask = (data["horizon"] == h).values
            if not mask.any():
                continue
            X = data.loc[mask, self.feature_names_ + cat].copy()
            for c in cat:
                X[c] = pd.Categorical(X[c], categories=self._categories_.get(c))
            preds[mask] = booster.predict(X)
        out["pred"] = preds
        return out


@dataclass
class SeasonalNaive:
    """Baseline: Vorhersage = Wert vor 12 Monaten (Saisonal-Naiv)."""

    season: int = 12

    def fit(self, data: pd.DataFrame, **kwargs) -> SeasonalNaive:
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data[["series", "date", "target_date", "horizon", "y"]].copy()
        # lag_12 ist im Feature-Set enthalten (Wert vor 12 Monaten).
        out["pred"] = data["lag_12"].values if "lag_12" in data.columns else np.nan
        return out


@dataclass
class Naive:
    """Baseline: Vorhersage = letzter beobachteter Wert."""

    def fit(self, data: pd.DataFrame, **kwargs) -> Naive:
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        out = data[["series", "date", "target_date", "horizon", "y"]].copy()
        out["pred"] = data["lag_1"].values if "lag_1" in data.columns else np.nan
        return out


def direct_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizons: tuple[int, ...],
    config: FeatureConfig | None = None,
    **params,
) -> pd.DataFrame:
    """Convenience: DirectLGBM fit + predict in einem Aufruf."""
    model = DirectLGBM(horizons=horizons, params=params)
    model.fit(train, config=config)
    return model.predict(test)
