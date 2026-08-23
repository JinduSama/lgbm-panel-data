"""
Gemeinsame Helfer für die Studien-Skripte (E1-E5).

Alle Ausgaben landen unter ``reports/``:
    reports/assets/*.png   Diagramme
    reports/results/*.json Kennzahlen (roh, für den Report-Builder)
"""

from __future__ import annotations

import base64
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reports" / "assets"
RESULTS = ROOT / "reports" / "results"

MODELS = ["lgbm", "seasonal_naive", "naive"]
MODEL_LABELS = {
    "lgbm": "LightGBM (global, direct)",
    "seasonal_naive": "Seasonal Naive",
    "naive": "Naive",
}
MODEL_COLORS = {"lgbm": "#2c7fb8", "seasonal_naive": "#f4a261", "naive": "#9d9d9d"}


def save_fig(fig: plt.Figure, name: str) -> str:
    """Speichert eine Figur unter reports/assets/<name>.png und liefert den Pfad."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out.relative_to(ROOT))


def save_result(name: str, payload: dict) -> None:
    """Schreibt Kennzahlen als JSON nach reports/results/<name>.json.

    Stamppt Reproduzierbarkeit-Metadaten (Git-SHA falls verfuegbar,
    UTC-Zeitstempel) in das Payload unter ``_meta``."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    import subprocess

    meta: dict = {"created_utc": pd.Timestamp.utcnow().isoformat()}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if sha:
            meta["git_sha"] = sha
    except Exception:
        pass
    RESULTS.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float)):
            return float(obj)
        if obj is None or isinstance(obj, str):
            return obj
        raise TypeError(f"nicht serialisierbar: {type(obj)}")

    stamped = dict(payload)
    stamped["_meta"] = meta
    (RESULTS / f"{name}.json").write_text(json.dumps(_clean(stamped), indent=2))


def metrics_dict(metrics: pd.DataFrame) -> dict:
    """model -> {horizon_str: {metrik: wert}} fuer JSON-Export."""
    out = {}
    for model, grp in metrics.groupby("model"):
        out[model] = {
            str(int(h)): row
            for h, row in grp.set_index("horizon")[["n", "mae", "rmse", "smape", "dir_acc"]]
            .to_dict("index")
            .items()
        }
    return out


def metrics_pivot(metrics: pd.DataFrame, value: str = "mae") -> pd.DataFrame:
    """model x horizon -> value."""
    return metrics.pivot(index="horizon", columns="model", values=value).sort_index()


def b64_image(rel_path: str) -> str:
    """PNG-Datei als Base64-Data-URI (fuer den selbstenthaltenen HTML-Report)."""
    data = (ROOT / rel_path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()
