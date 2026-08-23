"""
Datensatz-Ladung für LGBM Panel-Forecasting.

Lädt synthetische Panel-Daten oder echte M4-Monatsdaten (Long-Format).

Verwendung:
    from lgbm_panel.data.load import load_dataset
    df = load_dataset("synthetic")
    df = load_dataset("m4", n_series=500)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from .generate_synthetic import make_panel

DATA_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[2]


def load_dataset(name: str = "synthetic", **kwargs) -> pd.DataFrame:
    """
    Lädt einen Datensatz als Long-Format-Panel.

    Parameters
    ----------
    name : str
        ``"synthetic"`` oder ``"m4"``.
    **kwargs
        Zusatzparameter (z.B. ``n_series`` für M4-Sampling).

    Returns
    -------
    pd.DataFrame
        Spalten: ``series``, ``date``, ``value``.
    """
    if name == "synthetic":
        return make_panel(**kwargs)
    if name == "m4":
        return _load_m4(**kwargs)
    raise ValueError(f"Unbekannter Datensatz: {name!r}")


def _parse_start_months(raw: pd.Series, n_obs: pd.Series | None = None) -> pd.Series:
    """
    M4-Startdatum-Strings -> korrigierte Monatsordinals (int64).

    Zwei Formate in m4-info.csv:
      "01-01-79 12:00"  (DD-MM-YY, zweistelliges Jahr) und
      "1879-03-01 12:00:00" (ISO, z.B. Demographic-Serien).

    Zweistellige Jahre werden von Pandas nach 1969-2068 pivotiert - aber ein
    erheblicher Teil der langen M4-Monatsserien beginnt im 18./19. Jahrhundert
    ("31-01-90" = Januar 1790, nicht 1990). Die Korrektur ist datengetrieben:
    gewaehlt wird das groesste Jahrhundert-Rueckverschiebung k >= 0, sodass
    Start + Beobachtungsanzahl innerhalb des plausiblen Datenhorizonts
    (<= Ende 2030) liegt. Das disambiguiert 19xx gegen 18xx/17xx eindeutig,
    weil nur eine Dekade gleichzeitig zu kurzer UND zu langer Verlauf passt.
    """
    s = raw.astype(str).str.strip().str.split().str[0]
    as_dash = pd.to_datetime(s, format="%d-%m-%y", errors="coerce")
    as_iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    parsed = as_dash.fillna(as_iso)
    if parsed.isna().any():
        bad = sorted(raw[parsed.isna()].unique())
        raise ValueError(f"Unparsbare M4-Startdaten: {bad[:5]}")
    ordinals = parsed.dt.to_period("M").astype("int64")
    if n_obs is None:
        return ordinals
    horizon = pd.Period("2030-12", freq="M").ordinal
    overshoot = ordinals + n_obs.astype("int64") - horizon
    # k = ceil(overshoot / 1200), aber nie negativ (keine Vorwaerts-Schubse).
    k = (-(-overshoot.clip(lower=0) // 1200)).astype("int64")
    return ordinals - 1200 * k


def _load_m4(n_series: int | None = None, seed: int = 42) -> pd.DataFrame:
    """
    Laedt M4-Monatsserien im Long-Format (series, date, value).

    Der 91-MB-Roherstling wird einmalig in einen Parquet-Cache unter
    ``.cache/`` uebersetzt (Schluessel: Groesse + mtime der CSV); Sampling
    passiert danach im Speicher und bleibt bei identischem ``seed``
    reproduzierbar.
    """
    train_path = REPO_ROOT / "monthly-train.csv"
    info_path = REPO_ROOT / "m4-info.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"M4-Monatsdaten nicht gefunden: {train_path} (siehe README, Abschnitt Daten)"
        )

    cache_dir = REPO_ROOT / ".cache"
    stat = train_path.stat()
    cache_path = cache_dir / f"v2_m4_monthly_{stat.st_size}_{int(stat.st_mtime)}.parquet"

    if cache_path.exists():
        out = pd.read_parquet(cache_path)
    else:
        wide = pd.read_csv(train_path)
        info = pd.read_csv(info_path)
        id_col = wide.columns[0]  # "V1" -> Serien-ID
        long = wide.melt(id_vars=[id_col], var_name="col", value_name="value")
        long = long.rename(columns={id_col: "series"})
        long["step"] = long["col"].str.removeprefix("V").astype(int)
        long = long.dropna(subset=["value"]).sort_values(["series", "step"])

        # Startmonat JE Serie aus der Info-Datei (eindeutige Tabelle) plus
        # Beobachtungsanzahl fuer die Jahrhundert-Disambiguierung.
        start_dates = info.set_index("M4id")["StartingDate"]
        ids = long["series"].unique()
        starts_raw = pd.Series(ids, index=ids).map(start_dates)
        n_obs = long.groupby("series")["step"].max().reindex(ids).astype("int64")
        start_ord = _parse_start_months(starts_raw, n_obs).to_numpy()
        step_offset = (
            long["series"].map(pd.Series(start_ord, index=starts_raw.index)).to_numpy("int64")
            + (long["step"].to_numpy() - 1)
        )
        # Monats-Arithmetik in numpy; Zuweisung POSITIONAL (DatetimeIndex),
        # weil der Frame-Index nach sort_values nicht kontiguous ist!
        months = np.datetime64("1970-01", "M") + step_offset
        dates = pd.DatetimeIndex(months.astype("datetime64[M]").astype("datetime64[ns]"))
        long["date"] = dates + pd.offsets.MonthEnd(0)

        out = long[["series", "date", "value"]].reset_index(drop=True)
        cache_dir.mkdir(exist_ok=True)
        out.to_parquet(cache_path, index=False)

    if n_series is not None and n_series < out["series"].nunique():
        rng = np.random.default_rng(seed)
        sample = rng.choice(out["series"].unique(), size=n_series, replace=False)
        out = out[out["series"].isin(sample)].reset_index(drop=True)

    return out


if __name__ == "__main__":
    df = load_dataset("synthetic")
    print(f"Zeilen: {df.shape[0]}, Serien: {df['series'].nunique()}")
    print(df.head())
