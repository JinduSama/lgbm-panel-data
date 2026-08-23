"""
E7 - Zeitreihen-Galerie: echte Verlaeufe mit Prognosen.

Metriken komprimieren; hier sieht man die Muster:
(a) M4-Echtbeispiele: 6 reale Monatsserien, Prognose der letzten 18 Monate
    aus expanding Origin (globales LGBM auf Levels vs. Seasonal-Naive).
(b) Synthetische Regime-Beispiele: Raster aus E6-Trend-Regime x Saison-
    staerke (ohne / schwach / stark), je Zelle eine Beispielserie mit allen
    vier Varianten (Levels / SNaive / rekursiv Log-Diff / direkt Log-Diff).
    Echte Unternehmensdaten haben oft keine oder nur schwache Saison -
    starke Saisonalitaet allein waere ein unrealistischer Ausschnitt.

Alle Modelle trainieren ausschliesslich auf Daten vor dem Origin; die
geplottete Zukunft ist Wahrheit.
"""

from __future__ import annotations

import e6_levels_vs_logdiff as e6
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import save_fig
from lgbm_panel.data import load_dataset
from lgbm_panel.features import build_supervised
from lgbm_panel.strategies import DirectLGBM

H = 18

ALL_H = tuple(range(1, H + 1))  # Monatsaufloesung: 18 Punkte je Kurve, nicht 1


# --------------------------------------------------------------------------- #
# Hilfen: Varianten-Prognosen ab einem Origin fuer ausgewaehlte Serien.
# --------------------------------------------------------------------------- #
def _pad_panel(df: pd.DataFrame, origin: pd.Timestamp, value_cols: tuple[str, ...]) -> pd.DataFrame:
    """Panel um H Monate verlaengern (letzter Wert gehalten), damit
    Cutoff=Origin-Zeilen entstehen."""
    pad_dates = pd.date_range(origin + pd.DateOffset(months=1), periods=H, freq="MS")
    last = df.sort_values("date").groupby("series").tail(1).set_index("series")
    pads = []
    for s, row in last.iterrows():
        p = pd.DataFrame({"date": pad_dates})
        p.insert(0, "series", s)
        for c in value_cols:
            p[c] = row[c]
        pads.append(p)
    return pd.concat([df, *pads], ignore_index=True)


def _rows_at_origin(sup: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    rows = sup[sup["date"] == origin].copy()
    assert not rows.empty, f"keine Zeilen am Origin {origin}"
    return rows


def variant_forecasts(
    raw: pd.DataFrame, origin: pd.Timestamp, series_ids: list[str]
) -> dict[str, pd.DataFrame]:
    """model -> DataFrame(series, target_date, pred) fuer H Monate ab Origin."""
    out: dict[str, pd.DataFrame] = {}
    targets = pd.date_range(origin + pd.DateOffset(months=1), periods=H, freq="MS")

    # --- direkt auf Levels --------------------------------------------------
    padded = _pad_panel(raw, origin, ("value",))
    sup = build_supervised(padded, horizons=ALL_H, config=e6.DIRECT_CFG)
    rows = _rows_at_origin(sup[sup["series"].isin(series_ids)], origin)
    m_lvl = DirectLGBM(horizons=ALL_H, categorical=("series",)).fit(
        sup[sup["target_date"] <= origin], config=e6.DIRECT_CFG, num_boost_round=300
    )
    p = m_lvl.predict(rows)
    out["direct_level"] = p[["series", "target_date", "pred"]].rename(
        columns={"pred": "level_pred"}
    )

    # --- direkt auf Log-Diffs -------------------------------------------------
    log_df = raw.assign(value=np.log(raw["value"].clip(lower=1e-9)))
    log_padded = _pad_panel(log_df, origin, ("value",))
    supl = build_supervised(log_padded, horizons=ALL_H, config=e6.DIRECT_CFG)
    supl = supl.merge(
        log_df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
        on=["series", "date"],
        how="left",
    )
    rows_l = _rows_at_origin(supl[supl["series"].isin(series_ids)], origin)
    tr = supl[supl["target_date"] <= origin].dropna(subset=["y_ref"])
    tr = tr.assign(y_change=tr["y"] - tr["y_ref"])
    m_ld = DirectLGBM(horizons=ALL_H).fit(
        tr.assign(y=tr["y_change"]), config=e6.DIRECT_CFG, num_boost_round=300
    )
    pl = m_ld.predict(rows_l)
    pl["level_pred"] = np.exp(pl["y_ref"] + pl["pred"])
    out["direct_logdiff"] = pl[["series", "target_date", "level_pred"]]

    # --- rekursiv auf Log-Diffs ----------------------------------------------
    sup1 = (
        build_supervised(log_df, horizons=(1,), config=e6.RCFG)
        .merge(
            log_df[["series", "date", "value"]].rename(columns={"value": "y_ref"}),
            on=["series", "date"],
            how="left",
        )
        .dropna(subset=["y_ref"])
    )
    rec = DirectLGBM(horizons=(1,), categorical=()).fit(
        sup1.assign(y=sup1["y"] - sup1["y_ref"]), config=e6.RCFG, num_boost_round=300
    )
    feat_cols = list(e6._row_from_history(np.zeros(max(e6.R_LAGS)), origin))
    rec_rows = []
    lookup = log_df.set_index(["series", "date"])["value"]
    lvl_lookup = raw.set_index(["series", "date"])["value"]
    for s in series_ids:
        hist = [float(v) for k, v in lookup.loc[s].items() if k <= origin]
        v_end = hist[-1]
        cur, changes = origin, []
        for h in range(1, H + 1):
            row = e6._row_from_history(np.asarray(hist), cur)
            d = float(rec.models[1].predict(pd.DataFrame([row])[feat_cols])[0])
            changes.append(d)
            hist.append(v_end + sum(changes))
            cur = cur + pd.DateOffset(months=1)
            rec_rows.append(
                {
                    "series": s,
                    "target_date": targets[h - 1],
                    "level_pred": float(np.exp(v_end + sum(changes[:h]))),
                }
            )
        del hist  # noqa - Lesbarkeit: hist wird pro Serie neu aufgebaut
    out["recursive_logdiff"] = pd.DataFrame(rec_rows)

    # --- seasonal naive --------------------------------------------------------
    sn = []
    for s in series_ids:
        vals = lvl_lookup.loc[s]
        for tgt in targets:
            key = tgt - pd.DateOffset(months=12)
            if key in vals.index:
                sn.append({"series": s, "target_date": tgt, "level_pred": float(vals[key])})
    out["seasonal_naive"] = pd.DataFrame(sn)

    return out


MODEL_STYLE = {
    "direct_level": ("Direkt Levels", "#555555", "--"),
    "seasonal_naive": ("Seasonal Naive", "#f4a261", "-."),
    "direct_logdiff": ("Direkt Log-Diff", "#00798c", ":"),
}


def _plot_series(
    ax,
    raw: pd.DataFrame,
    s: str,
    origin: pd.Timestamp,
    preds: dict[str, pd.DataFrame],
    title: str,
    show_legend: bool,
    hist_months: int | None = None,
):
    """Eine Serie mit Historie, Wahrheit und allen Varianten-Prognosen."""
    hist_all = raw[(raw["series"] == s) & (raw["date"] <= origin)].sort_values("date")
    hist = hist_all.tail(hist_months) if hist_months else hist_all
    future = raw[(raw["series"] == s) & (raw["date"] > origin)].sort_values("date")
    ax.plot(hist["date"], hist["value"], color="#111111", lw=1.1, label="Historie")
    ax.plot(
        future["date"],
        future["value"],
        color="#111111",
        lw=2.6,
        alpha=0.85,
        label="Wahrheit",
        zorder=3,
    )
    for model, (label, color, ls) in MODEL_STYLE.items():
        p = preds.get(model)
        if p is None or p.empty:
            continue
        pp = p[p["series"] == s].sort_values("target_date")
        ax.plot(
            pp["target_date"], pp["level_pred"], color=color, ls=ls, lw=2.0, label=label, zorder=4
        )
    ax.axvline(origin, color="#888888", ls=":", lw=0.9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.5)
    if show_legend:
        ax.legend(fontsize=7, frameon=False)


def run() -> None:
    # ---------------- (a) M4-Echtbeispiele -----------------------------------
    m4 = load_dataset("m4", n_series=400)
    lengths = m4.groupby("series")["date"].count()
    picks = [s for s in lengths.sort_values(ascending=False).index if lengths[s] >= 120][:6]
    origin = m4[m4["series"].isin(picks)]["date"].max() - pd.DateOffset(months=H)
    preds_m4 = variant_forecasts(m4, origin, picks)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, s in zip(axes.ravel(), picks, strict=True):
        _plot_series(
            ax, m4, s, origin, preds_m4, f"M4 {s}", show_legend=(s == picks[0]), hist_months=120
        )
    handles, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle(
        f"E7a: M4-Monatsserien - Prognose der letzten {H} Monate ab {origin.date()}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_fig(fig, "e7_m4_examples")

    # ---------------- (b) Regime x Saisonalitaet ------------------------------
    season_levels = {"ohne": (0.0, 0.0), "schwach": (4.0, 8.0), "stark": (15.0, 35.0)}
    regimes = list({**e6.REGIMES, "stationaer": {"kind": "stationary"}}.items())
    regimes = list(e6.REGIMES.items())
    fig, axes = plt.subplots(len(regimes), len(season_levels), figsize=(16, 3.0 * len(regimes)))
    for r, (name, regime) in enumerate(regimes):
        for c, (sname, amp) in enumerate(season_levels.items()):
            raw = e6._panel(
                regime,
                seed=40 + c,
                season_amp=amp,
                n_series=10,
                rel_noise=0.03,
                spike_prob=0.05,  # Realismus: keine sauberen Funktionen
            )
            s = sorted(raw["series"].unique())[0]
            origin = raw["date"].max() - pd.DateOffset(months=H)
            preds = variant_forecasts(raw, origin, [s])
            _plot_series(
                axes[r][c],
                raw,
                s,
                origin,
                preds,
                f"{name} | Saison {sname}",
                show_legend=False,
            )
    handles, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(
        f"E7b: Trend-Regime × Saisonalität - je eine Beispielserie, "
        f"{H}-Monats-Prognosen aller vier Varianten",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_fig(fig, "e7_regime_examples")

    print("E7-Galerie fertig.")


if __name__ == "__main__":
    run()
