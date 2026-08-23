"""lgbm_panel.data - Datensatz-Verwaltung für Panel-Forecasting."""

from .generate_synthetic import make_panel
from .load import load_dataset

__all__ = ["make_panel", "load_dataset"]
