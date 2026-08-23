"""
E10 - Was treibt die Prognose? TreeSHAP-Treiberanalyse mit bekanntem DGP.

E3/E4 haben mit Ablation und Intervention gezeigt, dass Prognoseguete und
Kausalitaet verschiedene Dinge sind. Diese Studie geht einen Schritt weiter:
Wir lassen LightGBM selbst erklaeren, welche Eingaben die Prognose tragen -
und pruefen die Erklaerungen gegen die WAHRE Dynamik, weil wir das DGP
kennen (60 Serien x 132 Monate, DGP wie E3):

    x_t ~ |AR(1)|, phi=0.7            (kausaler Treiber, z.B. Budget)
    y_t = level + beta*x_{t-1} + Saison + Trend + Rauschen
    beta pro Serie ~ U(1.8, 2.6)

Protokoll:
    - Ein globales direktes LGBM (Horizonte 1..18), trainiert auf Zielen
      bis Monat 114; SHAP auf den ungesehenen letzten 18 Monaten
      (produktionstreu: eingefrorenes Modell wird erklaert).
    - TreeSHAP nativ via booster.predict(pred_contrib=True) - exakt,
      additiv: pred = base + sum(SHAP).

Vier Fragen:
    A) Budget: Wie verteilt mean|SHAP| die Erklærung auf Feature-Familien
       (Lags/Rolling/Kalender/Treiber/Entitaet) - im Vergleich zu Gain-
       Importance und zum wahren Signalbudget aus dem DGP?
    B) Recovery: Rekonstruiert SHAP(x) den kausalen Koeffizienten?
       h=1: Steigung von SHAP(x) vs x sollte beta je Serie treffen.
       h>1: bester erreichbarer Proxy ist x_t -> erwartete Steigung
       beta*phi^(h-1) (Signal-Zerfall des AR(1)-Treibers).
    C) Horizont-Profil: Welche Familien tragen bei h=1 vs h=18?
    D) Revisionen: Erklaert die SHAP-Differenz zweier Origins (gleicher
       Zielmonat) die Forecast-Aenderung - und folgt die Attribution der
       Informationsbewegung (Treiber bewegt vs ruhig)?
"""

from __future__ import annotations

import e3_feature_ablation as e3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import save_fig, save_result

from lgbm_panel.features import FeatureConfig, build_supervised
from lgbm_panel.strategies import DirectLGBM

N_SERIES = 60
N_PERIODS = 168
HORIZONS = tuple(range(1, 19))
TRAIN_END_IDX = 113  # letzte Zielmonats-Index im Training
PHI = 0.7  # AR(1)-Koeffizient des Treibers
FAMILY_ORDER = [
    "Target-Lags",
    "Rolling-Stats",
    "Saison-Diffs",
    "Kalender",
    "Treiber x",
    "Entit\u00e4t",
]
FAMILY_COLORS = {
    "Target-Lags": "#2c7fb8",
    "Rolling-Stats": "#7fcdbb",
    "Saison-Diffs": "#41ab5d",
    "Kalender": "#f4a261",
    "Treiber x": "#d1495b",
    "Entit\u00e4t": "#9d9d9d",
}


# --------------------------------------------------------------------- #
# Welt mit bekannten Komponenten
# --------------------------------------------------------------------- #
def make_world() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel + Komponenten-Tabelle (gleiche DGP wie e3.make_driven_panel)."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2015-01-01", periods=N_PERIODS, freq="MS")
    frames, meta = [], []
    for s in range(N_SERIES):
        t = np.arange(N_PERIODS, dtype=float)
        phase = rng.uniform(0, 2 * np.pi)
        beta = rng.uniform(1.8, 2.6)
        level = rng.uniform(80, 140)

        x = np.zeros(N_PERIODS)
        shocks = rng.normal(0, 6.0, N_PERIODS)
        for i in range(1, N_PERIODS):
            x[i] = PHI * x[i - 1] + shocks[i]
        x = np.abs(x)

        season = 25.0 * np.sin(2 * np.pi * t / 12.0 + phase)
        trend = 0.35 * t
        noise = rng.normal(0, 8.0, N_PERIODS)
        driver = beta * np.roll(x, 1)
        y = level + driver + season + trend + noise
        y[0] = level + season[0] + noise[0]

        frames.append(pd.DataFrame({"series": f"S{s:03d}", "date": dates, "value": y, "x": x}))
        meta.append(
            pd.DataFrame(
                {
                    "series": f"S{s:03d}",
                    "date": dates,
                    "driver": driver,
                    "season": season,
                    "trend": trend,
                    "beta": beta,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), pd.concat(meta, ignore_index=True)


# --------------------------------------------------------------------- #
# TreeSHAP-Helfer
# --------------------------------------------------------------------- #
def shap_table(model: DirectLGBM, rows: pd.DataFrame) -> pd.DataFrame:
    """Exakte TreeSHAP-Werte je Zeile (nativ via pred_contrib=True)."""
    cat = list(model.categorical)
    frames = []
    for h in sorted(model.models):
        sub = rows[rows["horizon"] == h]
        if sub.empty:
            continue
        X = sub[model.feature_names_ + cat].copy()
        for c in cat:
            X[c] = pd.Categorical(X[c], categories=model._categories_[c])
        contrib = model.models[h].predict(X, pred_contrib=True)
        names = ["f_series" if n == "series" else n for n in model.models[h].feature_name()]
        sv = pd.DataFrame(contrib[:, : len(names)], columns=names, index=sub.index)
        out = sub[["series", "date", "target_date", "y", "horizon"]].join(sv)
        out["base"] = contrib[:, len(names)]
        out["x_feat"] = X["x"].to_numpy()
        frames.append(out)
    res = pd.concat(frames, ignore_index=True)
    res["pred_shap"] = res["base"] + res[names].sum(axis=1)
    return res


ENTITY = "Entit\u00e4t"


def fam_of(feature: str) -> str:
    """Familie je Feature; die kategoriale Entitaet eigens benannt."""
    if feature in ("series", "f_series"):
        return ENTITY
    return e3.family_of(feature)


def family_shares(sv_cols: pd.DataFrame) -> dict[str, float]:
    """mean|SHAP| je Feature -> Anteil je Familie."""
    mean_abs = sv_cols.abs().mean()
    fam = (
        pd.Series({f: mean_abs.get(f, 0.0) for f in mean_abs.index})
        .groupby([fam_of(c) for c in mean_abs.index])
        .sum()
    )
    total = fam.sum()
    return {k: float(v / total) for k, v in fam.items()}


def gain_shares(model: DirectLGBM) -> dict[str, float]:
    """Gain-Importance ueber alle Horizont-Booster -> Familienanteile."""
    acc: dict[str, float] = {}
    for booster in model.models.values():
        gains = booster.feature_importance(importance_type="gain")
        for name, g in zip(booster.feature_name(), gains, strict=True):
            acc[fam_of(name)] = acc.get(fam_of(name), 0.0) + float(g)
    total = sum(acc.values())
    return {k: v / total for k, v in acc.items()}


def true_signal_budget(components: pd.DataFrame) -> dict[str, float]:
    """Wahres Signalbudget: mittlere within-series Std der DGP-Komponenten."""
    stds = components.groupby("series")[["driver", "season", "trend"]].std().mean()
    total = stds.sum()
    return {
        "Treiber x": float(stds["driver"] / total),
        "Saison (wahr)": float(stds["season"] / total),
        "Trend (wahr)": float(stds["trend"] / total),
    }


PROFILE_HORIZONS = (1, 2, 3, 6, 12, 18)
C0_IDX, C1_IDX = 104, 105  # die zwei Origins fuer die Revisions-Analyse


def _months(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + b.month - a.month


def fig_budget(shares_shap: dict, shares_gain: dict, truth: dict) -> None:
    """A) Drei Budgets im Vergleich: SHAP vs Gain vs wahres DGP-Signal."""
    rows: list[tuple[str, dict[str, float | None]]] = []
    for fam in FAMILY_ORDER:
        rows.append(
            (
                fam,
                {
                    "shap": shares_shap.get(fam),
                    "gain": shares_gain.get(fam),
                    "wahr": truth.get(fam),
                },
            )
        )
    for key in ("Saison (wahr)", "Trend (wahr)"):
        rows.append((key, {"shap": None, "gain": None, "wahr": truth.get(key)}))

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = {"shap": "#2c7fb8", "gain": "#f4a261", "wahr": "#41ab5d"}
    labels = {"shap": "mean |SHAP|", "gain": "Gain-Importance", "wahr": "Wahres DGP-Signal"}
    y = np.arange(len(rows))[::-1] * 1.0
    bh = 0.26
    for off, kind in ((bh, "shap"), (0.0, "gain"), (-bh, "wahr")):
        vals = [r[1][kind] for r in rows]
        pos = [yi + off for yi, v in zip(y, vals, strict=True) if v is not None]
        vv = [v for v in vals if v is not None]
        ax.barh(pos, vv, height=bh * 0.9, color=colors[kind], label=labels[kind])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Anteil an der totalen Erkl\u00e4rung")
    ax.set_title("E10 A: Erkl\u00e4rungs-Budget \u2013 was das Modell nennt vs. was wirklich wirkt")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.margins(y=0.02)
    save_fig(fig, "e10_family_budget")


def fig_recovery(decay: list[dict], slopes: pd.Series, betas: pd.Series) -> tuple[float, float]:
    """B) Koeffizienten-Recovery: Streigung je Serie + Zerfall ueber h."""
    corr = float(np.corrcoef(slopes.reindex(betas.index), betas)[0, 1])
    mae = float((slopes.reindex(betas.index) - betas).abs().mean())
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5))
    hs = [d["h"] for d in decay]
    ax_a.plot(
        hs, [d["slope"] for d in decay], "o-", color="#2c7fb8", label="Gemessene SHAP-Steigung"
    )
    ax_a.plot(
        hs,
        [d["ref"] for d in decay],
        "--",
        color="#d1495b",
        label=r"Referenz $\bar{\beta}\cdot\phi^{\,h-1}$ ($\phi=0{,}7$)",
    )
    ax_a.set_xlabel("Horizont h (Monate)")
    ax_a.set_ylabel("Steigung von SHAP(x) gegen x")
    ax_a.set_title("Treiber-Signal zerf\u00e4llt wie die AR(1)-Pr\u00e4diktion")
    ax_a.legend(fontsize=9, frameon=False)

    ax_b.scatter(betas, slopes.reindex(betas.index), s=28, alpha=0.75, color="#2c7fb8")
    lim = [min(betas.min(), slopes.min()) - 0.15, max(betas.max(), slopes.max()) + 0.15]
    ax_b.plot(lim, lim, "--", color="#9d9d9d", lw=1, label="Perfekte Recovery (45\u00b0)")
    ax_b.set_xlabel(r"Wahres $\beta$ (DGP)")
    ax_b.set_ylabel(r"Wiederhergestelltes $\beta$ (SHAP-Steigung, h=1)")
    ax_b.set_title(
        f"h=1: SHAP rekonstruiert den kausalen Koeffizienten\n(r={corr:.2f}, MAE={mae:.2f})"
    )
    ax_b.legend(fontsize=9, frameon=False)
    fig.suptitle("E10 B: Rekonstruktion des wahren Treiber-Effekts", y=1.02)
    save_fig(fig, "e10_recovery")
    return corr, mae


def fig_profiles(profiles: dict[int, dict[str, float]]) -> None:
    """C) Familienanteile je Horizont (gestapelt)."""
    fig, ax = plt.subplots(figsize=(10.5, 5))
    bottom = np.zeros(len(PROFILE_HORIZONS))
    xs = np.arange(len(PROFILE_HORIZONS))
    for fam in FAMILY_ORDER:
        vals = np.array([profiles[h].get(fam, 0.0) for h in PROFILE_HORIZONS])
        ax.bar(xs, vals, bottom=bottom, width=0.62, color=FAMILY_COLORS[fam], label=fam)
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([f"h={h}" for h in PROFILE_HORIZONS])
    ax.set_ylabel("Anteil an mean |SHAP|")
    ax.set_title("E10 C: Wer tr\u00e4gt die Prognose \u2013 kurze vs. lange Horizonte")
    ax.legend(fontsize=8, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save_fig(fig, "e10_horizon_profile")


def revision_pass(
    sh: pd.DataFrame,
    feat_cols: list[str],
    c0: pd.Timestamp,
    c1: pd.Timestamp,
    last_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """D) SHAP-Differenz zweier Origins fuer gemeinsame Zielmonate."""
    taus = pd.date_range(c1 + pd.offsets.MonthBegin(1), last_date, freq="MS")
    recs = []
    for tau in taus:
        h0, h1 = _months(c0, tau), _months(c1, tau)
        a = sh[(sh["date"] == c0) & (sh["horizon"] == h0)].set_index("series")
        b = sh[(sh["date"] == c1) & (sh["horizon"] == h1)].set_index("series")
        idx = a.index.intersection(b.index)
        if idx.empty:
            continue
        dsv = b.loc[idx, feat_cols] - a.loc[idx, feat_cols]
        row: dict = {"tau": tau}
        fams = sorted({fam_of(c) for c in feat_cols})
        for fam in fams:
            cols = [c for c in feat_cols if fam_of(c) == fam]
            contrib = dsv[cols].sum(axis=1)
            row[fam] = float(contrib.mean())
            row[f"{fam}_absmean"] = float(contrib.abs().mean())
        row["dpred_mean"] = float((b.loc[idx, "pred_shap"] - a.loc[idx, "pred_shap"]).mean())
        recs.append(row)
    rev = pd.DataFrame(recs)
    fams = [c for c in rev.columns if not c.endswith("_absmean") and c not in ("tau", "dpred_mean")]
    abs_sum = rev[[f"{f}_absmean" for f in fams]].sum()
    share = {f: float(abs_sum[f"{f}_absmean"] / abs_sum.sum()) for f in fams}
    signed = {
        f: float(rev[f].abs().sum() / rev[[f"{g}_absmean" for g in fams]].sum().sum()) for f in fams
    }
    return rev, share, signed


def _example_revision(
    sh: pd.DataFrame,
    panel: pd.DataFrame,
    feat_cols: list[str],
    c0: pd.Timestamp,
    c1: pd.Timestamp,
    series: str,
) -> dict:
    """Einzelnes Serie-Beispiel: Familienbeitraege zur Revision je Zielmonat."""
    a = sh[(sh["date"] == c0) & (sh["series"] == series)].set_index(["target_date", "horizon"])
    b = sh[(sh["date"] == c1) & (sh["series"] == series)].set_index(["target_date", "horizon"])
    taus = sorted(set(a.index.get_level_values(0)) & set(b.index.get_level_values(0)))
    best_mag = -1.0
    out = {}
    for tau in taus:
        ra = a.xs(tau, level="target_date").iloc[0]
        rb = b.xs(tau, level="target_date").iloc[0]
        dsv = {c: float(rb[c] - ra[c]) for c in feat_cols}
        mag = sum(abs(v) for v in dsv.values())
        if mag > best_mag:
            best_mag = mag
            fam_contrib: dict[str, float] = {}
            for c, v in dsv.items():
                fam_contrib[fam_of(c)] = fam_contrib.get(fam_of(c), 0.0) + v
            out = {
                "tau": str(tau.date()),
                "families": fam_contrib,
                "dpred": float(rb["pred_shap"] - ra["pred_shap"]),
                "err_before": float(ra["y"] - ra["pred_shap"]),
                "err_after": float(rb["y"] - rb["pred_shap"]),
                "dx": _driver_delta(panel, series, c0, c1),
            }
    out["series"] = series
    return out


def _driver_delta(panel: pd.DataFrame, series: str, c0: pd.Timestamp, c1: pd.Timestamp) -> float:
    sub = panel[panel["series"] == series].set_index("date")["x"]
    return float(sub.loc[c1] - sub.loc[c0])


def fig_revision(share_abs: dict[str, float], move: dict, quiet: dict) -> None:
    """D) Revisionen: Aggregate + zwei gearbeitete Beispiele."""
    fig, (ax_agg, ax_move, ax_quiet) = plt.subplots(1, 3, figsize=(15.5, 5))

    fams = [f for f in FAMILY_ORDER if f in share_abs]
    vals = [share_abs[f] for f in fams]
    ax_agg.barh(np.arange(len(fams))[::-1], vals, color=[FAMILY_COLORS[f] for f in fams])
    ax_agg.set_yticks(np.arange(len(fams))[::-1])
    ax_agg.set_yticklabels(fams)
    ax_agg.set_xlabel("Anteil an |Revision|")
    ax_agg.set_title(
        "Wer erkl\u00e4rt Forecast-Revisionen?\n(mittel \u00fcber Serien & Zielmonate)"
    )

    for ax, ex, title in (
        (ax_move, move, "Treiber bewegt sich"),
        (ax_quiet, quiet, "Treiber ruhig"),
    ):
        fams_ex = sorted(ex["families"], key=lambda f: abs(ex["families"][f]), reverse=True)
        vv = [ex["families"][f] for f in fams_ex]
        ax.bar(
            np.arange(len(fams_ex)), vv, color=[FAMILY_COLORS.get(f, "#9d9d9d") for f in fams_ex]
        )
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xticks(np.arange(len(fams_ex)))
        ax.set_xticklabels([f.replace(" ", "\n") for f in fams_ex], fontsize=7.5)
        dp, dx = ex["dpred"], ex["dx"]
        eb, ea = ex["err_before"], ex["err_after"]
        ax.set_title(
            f"{title} \u2013 Serie {ex['series']}, Ziel {ex['tau']}\n"
            f"$\\Delta$x = {dx:+.2f}, $\\Delta$Prognose = {dp:+.2f}\n"
            f"Fehler: {eb:+.2f} $\\rightarrow$ {ea:+.2f}",
            fontsize=9.5,
        )


def run() -> dict:
    panel, comps = make_world()
    cfg = FeatureConfig(exog_cols=("x",))
    data = build_supervised(panel, horizons=HORIZONS, config=cfg)
    train_end = panel["date"].iloc[TRAIN_END_IDX]
    train = data[data["target_date"] <= train_end].copy()
    hold = data[data["target_date"] > train_end].copy()

    model = DirectLGBM(horizons=HORIZONS)
    model.fit(train, config=cfg, num_boost_round=400)
    feat_cols = list(model.feature_names_)
    sh = shap_table(model, hold)
    shap_feat_cols = feat_cols + ["f_series"]  # inkl. Entitaet-Block

    # Additivitaet: base + sum(SHAP) muss der Booster-Vorhersage entsprechen.
    sub1 = hold[hold["horizon"] == 1]
    X1 = sub1[feat_cols + ["series"]].copy()
    X1["series"] = pd.Categorical(X1["series"], categories=model._categories_["series"])
    direct = model.models[1].predict(X1)
    additivity_err = float(
        (sh.loc[sh["horizon"] == 1, "pred_shap"].to_numpy() - direct).__abs__().max()
    )

    # --- A) Budgets -----------------------------------------------------
    shares_shap = family_shares(sh[shap_feat_cols])
    shares_gain = gain_shares(model)
    truth = true_signal_budget(comps)
    fig_budget(shares_shap, shares_gain, truth)

    # --- B) Recovery ----------------------------------------------------
    betas = comps.groupby("series")["beta"].first()
    s1 = sh[sh["horizon"] == 1]
    slopes = s1.groupby("series").apply(
        lambda g: float(np.polyfit(g["x_feat"], g["x"], 1)[0]), include_groups=False
    )
    decay = []
    for h, grp in sh.groupby("horizon"):
        slope = float(np.polyfit(grp["x_feat"], grp["x"], 1)[0])
        decay.append(
            {"h": int(h), "slope": slope, "ref": float(betas.mean() * PHI ** (int(h) - 1))}
        )
    corr, beta_mae = fig_recovery(decay, slopes, betas)

    # --- C) Horizont-Profile -------------------------------------------
    profiles = {int(h): family_shares(grp[shap_feat_cols]) for h, grp in sh.groupby("horizon")}
    fig_profiles({h: profiles[h] for h in PROFILE_HORIZONS})

    # --- D) Revisionen --------------------------------------------------
    c0, c1 = panel["date"].iloc[C0_IDX], panel["date"].iloc[C1_IDX]
    rev, share_abs, signed = revision_pass(
        sh, shap_feat_cols, c0, c1, panel["date"].iloc[N_PERIODS - 1]
    )

    dx = {s: _driver_delta(panel, s, c0, c1) for s in panel["series"].unique()}
    ser_move = max(dx, key=lambda s: abs(dx[s]))
    ser_quiet = min(dx, key=lambda s: abs(dx[s]))
    ex_move = _example_revision(sh, panel, shap_feat_cols, c0, c1, ser_move)
    ex_quiet = _example_revision(sh, panel, shap_feat_cols, c0, c1, ser_quiet)
    fig_revision(share_abs, ex_move, ex_quiet)

    payload = {
        "setup": {
            "n_series": N_SERIES,
            "n_periods": N_PERIODS,
            "train_targets_until": str(train_end.date()),
            "holdout_months": N_PERIODS - 1 - TRAIN_END_IDX,
            "phi": PHI,
            "beta_range": [1.8, 2.6],
            "additivity_max_err": additivity_err,
            "n_holdout_rows": int(len(sh)),
        },
        "budget": {"shap": shares_shap, "gain": shares_gain, "truth": truth},
        "recovery": {"beta_corr_h1": corr, "beta_mae_h1": beta_mae, "slope_decay": decay},
        "profiles": {str(h): profiles[h] for h in PROFILE_HORIZONS},
        "revision": {
            "family_share_of_abs_revision": share_abs,
            "family_signed_mean": signed,
            "example_driver_moves": {"move": ex_move, "quiet": ex_quiet},
            "origins": [str(c0.date()), str(c1.date())],
        },
    }
    save_result("e10_shap_drivers", payload)
    return payload


if __name__ == "__main__":
    from pprint import pprint

    pprint(run())
