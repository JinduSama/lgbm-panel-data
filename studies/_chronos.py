"""
Gemeinsame Chronos-Arme fuer Studien (E14/E15).

Zwei Pipelines:
    amazon/chronos-bolt-base : univariate Null-Shot-Basis (keine Kovariaten)
    amazon/chronos-2         : universal ICL, unterstuetzt bekannte
                               Zukunftskovariaten via ``future_df``

Alle Arme liefern Fixed-Origin-Blockprognosen im selben Schema wie
e11_m4_best.SCHEMA; Monats-Matching ueber Ordinals (Jahr*12+Monat), damit
die Frequenz-Konvention der Pipelines keine Rolle spielt.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

_PIPELINE_CACHE: dict[str, object] = {}

CONTEXT_LIMIT = 512  # letzte Beobachtungen je Serie


def _month_ord(s: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(s) if not isinstance(s, pd.DatetimeIndex) else s
    return (idx.year * 12 + idx.month).to_numpy()


def _get_bolt(model_id: str = "amazon/chronos-bolt-base"):
    if model_id not in _PIPELINE_CACHE:
        from chronos import BaseChronosPipeline

        _PIPELINE_CACHE[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", dtype=torch.float32
        )
    return _PIPELINE_CACHE[model_id]


def _get_c2(model_id: str = "amazon/chronos-2"):
    if model_id not in _PIPELINE_CACHE:
        from chronos import Chronos2Pipeline

        _PIPELINE_CACHE[model_id] = Chronos2Pipeline.from_pretrained(
            model_id, device_map="cpu", dtype=torch.float32
        )
    return _PIPELINE_CACHE[model_id]


def _bucket_by_horizon(
    fc_ord: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    series_ids: np.ndarray,
    fe: pd.Series,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Prognose-Ordinals in Horizont-Buckets fe+h je Serie abbilden."""
    fe_dates = pd.Series(fe.reindex(series_ids).to_numpy(), index=series_ids)
    fe_ord = fe_dates.dt.year * 12 + fe_dates.dt.month
    frames = []
    for sid in series_ids:
        o = fc_ord[sid]
        p = preds[sid]
        tbl = pd.DataFrame({"tord": o, "pred": p}).set_index("tord")["pred"]
        want = int(fe_ord.loc[sid])
        rows = []
        for h in horizons:
            val = tbl.get(want + h, np.nan)
            rows.append({"horizon": h, "pred": val})
        f = pd.DataFrame(rows)
        f["series"] = sid
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def chronos_bolt_arm(
    df: pd.DataFrame,
    fold_end_map: dict[int, pd.Series],
    horizons: tuple[int, ...],
    step: int,
    n_folds: int,
    schema: list[str],
    model_id: str = "amazon/chronos-bolt-base",
    name: str = "chronos-bolt-base",
) -> pd.DataFrame:
    """Univariate Null-Shot-Blockprognose vom eigenen fold_end je Serie."""
    pipe = _get_bolt(model_id)
    lv = df.set_index(["series", "date"])["value"]
    hist_by_series = {s: g.sort_values("date")["value"].to_numpy(dtype=np.float32)
                      for s, g in df.groupby("series")}
    frames = []
    for i in range(1, n_folds + 1):
        fe = fold_end_map[i]
        series_ids = fe.index.to_numpy()
        contexts = [
            torch.tensor(hist_by_series[s][-CONTEXT_LIMIT:], dtype=torch.float32)
            for s in series_ids
        ]
        with torch.inference_mode():
            q, _mean = pipe.predict_quantiles(inputs=contexts, prediction_length=step)
        # q: [n_series, step, 9] -> Median ist Index 4 (Quantile 0.1..0.9)
        med = q[:, :, 4].numpy()
        ordinals = {
            s: np.arange(int(_month_ord(pd.DatetimeIndex([fe.loc[s]]))[0]) + 1,
                         int(_month_ord(pd.DatetimeIndex([fe.loc[s]]))[0]) + step + 1)
            for s in series_ids
        }
        preds = {s: med[k] for k, s in enumerate(series_ids)}
        b = _bucket_by_horizon(ordinals, preds, series_ids, fe, horizons)
        b["cutoff"] = b["series"].map(fe)
        offsets = b["horizon"].apply(lambda h: pd.DateOffset(months=h))
        b["target_date"] = (pd.Series(b["cutoff"].to_numpy()) + offsets).to_numpy()
        b["y"] = lv.reindex(
            pd.MultiIndex.from_arrays([b["series"], b["target_date"]])
        ).to_numpy()
        b["y_ref"] = lv.reindex(pd.MultiIndex.from_arrays([b["series"], b["cutoff"]])).to_numpy()
        b["model"] = name
        b["fold"] = i
        frames.append(b[schema])
    return pd.concat(frames, ignore_index=True)


def chronos2_arm(
    df: pd.DataFrame,
    fold_end_map: dict[int, pd.Series],
    horizons: tuple[int, ...],
    step: int,
    n_folds: int,
    schema: list[str],
    exog_col: str | None = None,
    model_id: str = "amazon/chronos-2",
    name: str = "chronos-2",
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Chronos-2 Blockprognose. Mit ``exog_col`` wird die Spalte als bekannte
    Zukunftskovariate uebergeben (Budget-Plan-Szenario): Historie + Zukunft
    werden dem Modell vollstaendig bereitgestellt.
    """
    pipe = _get_c2(model_id)
    lv = df.set_index(["series", "date"])["value"]

    def _norm(d: pd.Series | pd.Index) -> pd.Series:
        return pd.Series(pd.DatetimeIndex(d)).dt.to_period("M").dt.to_timestamp(how="start")

    frames = []
    cols = ["item_id", "timestamp", "target"] + ([exog_col] if exog_col else [])
    for i in range(1, n_folds + 1):
        fe = fold_end_map[i]
        d = df.merge(fe.rename("fe"), on="series", how="left")
        hist = d[d["date"] <= d["fe"]]
        # POSITIONAL (.to_numpy()): gemischte Indizes wuerden Zeilen
        # stumm vertauschen (klassische pandas Alignment-Falle).
        hist_long = pd.DataFrame({
            "item_id": hist["series"].to_numpy(),
            "timestamp": _norm(hist["date"]).to_numpy(),
            "target": hist["value"].to_numpy(),
        })
        if exog_col:
            hist_long[exog_col] = hist[exog_col].to_numpy()

        # Zukunftsfenster je Serie: fe+1 .. fe+step (Monatsanfaenge).
        fut_rows = []
        for s, g in hist_long.groupby("item_id", sort=False):
            last = g["timestamp"].max()
            fut_ts = pd.date_range(last + pd.offsets.MonthBegin(1), periods=step, freq="MS")
            f = pd.DataFrame({"item_id": s, "timestamp": fut_ts})
            if exog_col:
                vals = d[(d["series"] == s)][["date", exog_col]].sort_values("date")
                vord = dict(zip(_month_ord(vals["date"]), vals[exog_col].to_numpy(), strict=True))
                f[exog_col] = [(vord.get(ts.year * 12 + ts.month)) for ts in fut_ts]
            fut_rows.append(f)
        future_long = pd.concat(fut_rows, ignore_index=True)
        fut_cols = ["item_id", "timestamp"] + ([exog_col] if exog_col else [])
        out = pipe.predict_df(
            hist_long[cols],
            future_df=future_long[fut_cols],
            prediction_length=step,
            batch_size=batch_size,
            quantile_levels=(0.5,),
        )
        out = out.rename(columns={"item_id": "series", "timestamp": "ds"})
        preds_all: dict[str, np.ndarray] = {}
        ord_all: dict[str, np.ndarray] = {}
        for s, g in out.groupby("series"):
            order = np.argsort(g["ds"].to_numpy())
            preds_all[s] = g["0.5"].to_numpy()[order]
            ord_all[s] = _month_ord(g["ds"].iloc[order])
        series_ids = fe.index.to_numpy()
        b = _bucket_by_horizon(ord_all, preds_all, series_ids, fe, horizons)
        b["cutoff"] = b["series"].map(fe)
        offsets = b["horizon"].apply(lambda h: pd.DateOffset(months=h))
        b["target_date"] = (pd.Series(b["cutoff"].to_numpy()) + offsets).to_numpy()
        b["y"] = lv.reindex(
            pd.MultiIndex.from_arrays([b["series"], b["target_date"]])
        ).to_numpy()
        b["y_ref"] = lv.reindex(pd.MultiIndex.from_arrays([b["series"], b["cutoff"]])).to_numpy()
        b["model"] = name
        b["fold"] = i
        frames.append(b[schema])
    return pd.concat(frames, ignore_index=True)
