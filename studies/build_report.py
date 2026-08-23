"""
Baut den selbstenthaltenen HTML-Report aus den Studien-Ergebnissen.

Liest reports/results/*.json und bettet die PNGs aus reports/assets/
base64-kodiert ein -> eine einzige Datei ``reports/report.html``.

Der Report ist bewusst ausfuehrlich: Setup-Boxen je Studie, vollstaendige
Metrik-Tabellen, Lesefuehrer je Abbildung und Methodik-Anhang.
"""

from __future__ import annotations

import json

from _common import ROOT, b64_image

RESULTS = ROOT / "reports" / "results"

CSS = """
:root {
  --bg:#0f1420; --card:#171e2e; --ink:#e8ecf4; --muted:#9aa7bd;
  --accent:#4cc3ff; --good:#5ad19c; --bad:#ff7b72; --line:#28324a;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:16px/1.65 "Segoe UI",system-ui,-apple-system,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:48px 24px 96px; }
header.hero { padding:40px 0 8px; }
h1 { font-size:34px; margin:0 0 6px; letter-spacing:-0.5px; }
p.sub { color:var(--muted); margin:0 0 28px; }
h2 { font-size:24px; margin:56px 0 12px; padding-top:18px;
     border-top:1px solid var(--line); }
h3 { font-size:17px; margin:26px 0 8px; color:var(--accent); }
h4 { font-size:14px; margin:18px 0 6px; color:var(--muted);
     text-transform:uppercase; letter-spacing:0.06em; }
p, li { color:#c9d3e4; }
code { background:#1d2740; border:1px solid var(--line); padding:1px 6px;
       border-radius:6px; font-size:13px; color:#a8d5ff; }
pre { background:#121a2b; border:1px solid var(--line); border-radius:10px;
      padding:14px 16px; overflow-x:auto; font-size:13px; }
pre code { border:none; background:none; padding:0; }
.card { background:var(--card); border:1px solid var(--line);
        border-radius:14px; padding:20px 24px; margin:18px 0; }
.card.setup { font-size:14px; }
.card.setup b { color:var(--ink); }
.finding { border-left:4px solid var(--accent); }
.warn { border-left:4px solid var(--bad); }
.goodbox { border-left:4px solid var(--good); }
img.fig { max-width:100%; border-radius:10px; border:1px solid var(--line);
          margin:14px 0; background:#fff; }
.figread { color:var(--muted); font-size:13.5px; margin-top:-6px; }
table { width:100%; border-collapse:collapse; margin:14px 0; font-size:14px; }
th, td { padding:7px 11px; text-align:right; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; text-transform:uppercase;
     font-size:11px; letter-spacing:0.08em; }
td.hl { color:var(--good); font-weight:600; }
td.lo { color:var(--bad); }
tr.group td { color:var(--muted); font-size:12px; text-transform:uppercase;
              letter-spacing:0.06em; padding-top:14px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
        gap:14px; margin:22px 0; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:12px;
       padding:16px 18px; }
.kpi .v { font-size:24px; font-weight:700; color:var(--accent); }
.kpi .t { color:var(--muted); font-size:13px; margin-top:2px; }
details { margin:10px 0; }
summary { cursor:pointer; color:var(--accent); }
footer { margin-top:80px; color:var(--muted); font-size:13px;
         border-top:1px solid var(--line); padding-top:18px; }
nav.toc a { color:var(--accent); text-decoration:none; }
nav.toc li { margin:4px 0; }
"""


def load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig(name: str, alt: str, reading: str | None = None) -> str:
    rel = f"reports/assets/{name}.png"
    if not (ROOT / rel).exists():
        return f"<p><em>Figur {name} fehlt.</em></p>"
    extra = f'<p class="figread"><strong>Wie lesen?</strong> {reading}</p>' if reading else ""
    return f'<img class="fig" src="{b64_image(rel)}" alt="{alt}"/>{extra}'


def fmt(v: float | None, digits: int = 2) -> str:
    return f"{v:.{digits}f}" if v is not None else "-"


def cell(v: float | None, best: float | None, digits: int = 2) -> str:
    """Eine Tabellenzelle; Bestwert wird grueng hervorgehoben."""
    if v is None:
        return "<td>-</td>"
    cls = ' class="hl"' if best is not None and abs(v - best) < 1e-12 else ""
    return f"<td{cls}>{fmt(v, digits)}</td>"


def metrics_table(
    metrics: dict,
    values: tuple[str, ...] = ("mae",),
    digits: int = 2,
    lower_is_better: dict[str, bool] | None = None,
) -> str:
    """model -> {horizon -> {metric}} als eine Tabelle mit Metrik-Bloecken."""
    lib = lower_is_better or {}
    models = sorted(metrics)
    horizons = sorted({h for m in metrics.values() for h in m}, key=float)
    rows = []
    for value in values:
        rows.append(f'<tr class="group"><td colspan="{len(models) + 1}">{value.upper()}</td></tr>')
        for h in horizons:
            vals = [metrics[m].get(h, {}).get(value) for m in models]
            finite = [v for v in vals if v is not None]
            lib_ok = lib.get(value, True)
            best = (min(finite) if lib_ok else max(finite)) if finite else None
            cells = "".join(cell(v, best, digits) for v in vals)
            rows.append(f"<tr><td>h={h}</td>{cells}</tr>")
    head = "".join(f"<th>{m}</th>" for m in models)
    return (
        f"<table><thead><tr><td>Horizont</td>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def setup_box(**items: object) -> str:
    body = " &nbsp;·&nbsp; ".join(f"<b>{k}</b> {v}" for k, v in items.items())
    return f'<div class="card setup">{body}</div>'


def main() -> None:
    e1 = load("e1_scenarios")
    e2 = load("e2_data_prep")
    e3 = load("e3_feature_ablation")
    e4 = load("e4_causal")
    e5 = load("e5_m4")
    e6 = load("e6_levels_vs_logdiff")
    e8 = load("e8_combined")
    e9 = load("e9_tuning")
    e10 = load("e10_shap_drivers")
    e11 = load("e11_m4_best")
    e12 = load("e12_intervals")
    e13 = load("e13_objective_ablation")
    e14 = load("e14_chronos_m4")
    e15 = load("e15_chronos_exog")
    parts: list[str] = []

    # ------------------------------------------------------------------ TOC
    parts.append(
        "<h2>Inhalt</h2>"
        '<nav class="toc"><ol>'
        "<li><a href='#tldr'>Kernbefunde</a></li>"
        "<li><a href='#method'>Methodik im Detail</a></li>"
        "<li><a href='#gallery'>Zeitreihen-Galerie</a></li>"
        "<li><a href='#e1'>E1 &middot; Szenario-Raster</a></li>"
        "<li><a href='#e2'>E2 &middot; Datenaufbereitung</a></li>"
        "<li><a href='#e3'>E3 &middot; Feature-Ablation</a></li>"
        "<li><a href='#e4'>E4 &middot; Kausale Plausibilit&auml;t</a></li>"
        "<li><a href='#e5'>E5 &middot; M4-Benchmark</a></li>"
        "<li><a href='#e6'>E6 &middot; Level vs. Log-Diffs &amp; Rekursion</a></li>"
        "<li><a href='#e8'>E8 &middot; Alles zusammen</a></li>"
        "<li><a href='#e9'>E9 &middot; Hyperparameter-Tuning</a></li>"
        "<li><a href='#e10'>E10 &middot; Treiber-Attribution (TreeSHAP)</a></li>"
        "<li><a href='#e11'>E11 &middot; Beste Formulierung vs. lokal (M4)</a></li>"
        "<li><a href='#e12'>E12 &middot; Prognoseintervalle</a></li>"
        "<li><a href='#e13'>E13 &middot; Objective-Ablation</a></li>"
        "<li><a href='#e14'>E14 &middot; Foundation Models (M4)</a></li>"
        "<li><a href='#e15'>E15 &middot; Budgetpl&auml;ne als Kovariaten</a></li>"
        "<li><a href='#literatur'>Einordnung: Praxis &amp; Literatur</a></li>"
        "<li><a href='#synthese'>Gegen&uuml;berstellung</a></li>"
        "<li><a href='#takeaways'>Empfehlungen</a></li>"
        "<li><a href='#appendix'>Anhang &amp; Reproduktion</a></li>"
        "</ol></nav>"
    )

    # ------------------------------------------------------------------ TLDR
    e2_ratio = None
    if e2:
        s = e2["scenarios"]
        if "levels" in s and "log_seasdiff12" in s:
            a = s["levels"].get("12", {}).get("mae")
            b = s["log_seasdiff12"].get("12", {}).get("mae")
            if a and b:
                e2_ratio = a / b
    e6_ratio = None
    if e6:
        regimes = e6["metrics_on_levels"]
        stark = regimes.get("stark_trendend", {})
        r = stark.get("recursive_logdiff", {}).get("18", {}).get("mae")
        d = stark.get("direct_logdiff", {}).get("18", {}).get("mae")
        if r and d:
            e6_ratio = r / d
    e4_dir = None
    if e4:
        iv = {r["model"]: r["dir_acc"] for r in e4["intervention"]}
        e4_dir = iv.get("with_x_plan")

    e8_drop = None
    if e8:
        mm8 = e8["metrics_on_levels"]
        lv = mm8.get("levels", {}).get("18", {}).get("mae")
        lx = mm8.get("logdiff_x", {}).get("18", {}).get("mae")
        if lv and lx:
            e8_drop = 1 - lx / lv
    kpis = []
    if e9:
        imps = {k: v["improvement_pct"] for k, v in e9["scenarios"].items()}
        best = max(imps.values())
        kpis.append(
            (
                f"&le;&nbsp;{best:.0f}&nbsp;%",
                "Maximaler MAE-Gewinn durch Hyperparameter-Tuning &uuml;ber alle"
                " Szenarien - Formulierung &amp; Features schlagen Tuning (E9)",
            )
        )
    if e10:
        rec = e10["recovery"]
        h1 = next(d for d in rec["slope_decay"] if d["h"] == 1)
        ratio = 100 * h1["slope"] / h1["ref"]
        kpis.append(
            (
                f"{ratio:.0f}&nbsp;%",
                "SHAP-Steigung vs. wahrem Treiber-Koeffizient bei h=1: das Modell"
                " erkl&auml;rt seinen kausalen Kern selbst (E10)",
            )
        )
    if e2_ratio:
        kpis.append(
            (
                f"{e2_ratio:,.0f}×",
                "weniger Fehler durch Log-Saisondifferenzen statt Levels (h=12, E2)",
            )
        )
    if e6_ratio:
        kpis.append(
            (
                f"{e6_ratio:,.1f}×",
                "Rekursions-Strafe bei h=18 auf stabil exponentiellem Trend:"
                " rekursiv vs. direkt auf Log-Diffs (E6)",
            )
        )
    if e4_dir:
        kpis.append(
            (
                f"{100 * e4_dir:.0f} %",
                "Directional Accuracy nach Intervention - nur mit Treiber-Pfad (E4)",
            )
        )
    if e5:
        mase = e5["mase_overall"].get("lgbm")
        if mase:
            kpis.append(
                (
                    f"{mase:.2f}",
                    "MASE auf 400 echten M4-Serien - bester von drei Modellen, "
                    "aber &gt;1: die 1-Schritt-Referenz bleibt hart (E5)",
                )
            )
    if e8_drop:
        kpis.append(
            (
                f"&minus;{100 * e8_drop:.0f}&nbsp;%",
                "MAE bei h=18 im kombinierten DGP: Log-Diffs + Treiber statt roher Levels (E8)",
            )
        )
    if kpis:
        parts.append(
            '<h2 id="tldr">Kernbefunde</h2><div class="kpis">'
            + "".join(
                f'<div class="kpi"><div class="v">{v}</div><div class="t">{t}</div></div>'
                for v, t in kpis
            )
            + "</div>"
        )

    # ------------------------------------------------------------------ method
    parts.append("""
<h2 id="method">Methodik im Detail</h2>

<h3>Vom Panel zur Supervised-Tabelle</h3>
<div class="card">
<p>Jede Serie wird in Zeilen <code>(Serie, Cutoff t, Horizont h)</code> zerlegt.
F&uuml;r eine monatliche Serie mit 132 Beobachtungen und 5 Horizonten entstehen
so bis zu 5&times;132 Zeilen - das Panel-Learning nutzt alle Serien gleichzeitig
in <em>einem</em> globalen LightGBM.</p>
<p><strong>Features (Default-Konfiguration):</strong></p>
<ul>
<li>Target-Lags: <code>1, 2, 3, 6, 12, 13, 18, 24</code> (Wert zum Zeitpunkt t&minus;k)</li>
<li>Rolling-Stats: Fenster <code>3, 6, 12</code> &times; <code>mean, std, min, max</code></li>
<li>Saisondifferenzen: <code>1, 12</code></li>
<li>Kalender: <code>month</code> (des Cutoffs)</li>
<li>optional: exogene Treiber (Wert bei t), Szenario-Treiber (Wert bei Ziel&minus;j),
cross-sectionale Aggregate (Panel-Mittel/Std zum Zeitpunkt t)</li>
</ul>
<p><strong>Label:</strong> <code>y[t+h]</code>. Da nur Informationen mit
Zeitstempel &le; t in die Features gehen, ist Leakage per Konstruktion
ausgeschlossen. Trainingszeilen im Backtest werden zusaetzlich nach
<code>target_date &le; Fold-Ende</code> gefiltert - auch das <em>Ziel</em> muss
bereits beobachtet sein.</p>
</div>

<h3>Direct Multi-Horizon statt rekursiv</h3>
<div class="card">
<p>F&uuml;r jeden Horizont h wird ein eigenes Modell trainiert, das
t+h <em>direkt</em> vorhersagt. Vorteil: kein Fehler-Schneeball durch
Wiedereinspeisung eigener Prognosen (quantifiziert in <a href="#e6">E6</a>).
Kosten: ein Booster pro Horizont - bei 18 Horizonten und 300 B&auml;umen
immer noch Sekunden.</p>
</div>

<h3>Backtest-Design</h3>
<div class="card">
<p>Expanding Window mit je-Serie verankerten Folds: Fold k trainiert auf
allen Zielen bis Stichtag T<sub>k</sub> und testet auf die folgenden
<code>step</code> Monate (<code>step = max(Horizonte)</code>,
nicht-&uuml;berlappende Testfenster). Auf Panelen mit ungleichen Serienenden
(M4) liegt der Stichtag je Serie an deren eigenem Ende - jede Serie wird
ueber ihre volle Traegerbreite evaluiert:</p>
<pre><code>Historie:  |—————— Train ——————| Test |
Fold 1:    |—————— T1 —————————|  T1+step  |
Fold 2:    |—————————— T2 ————————|  T2+step  |
Fold 3:    |—————————————— T3 ———————|  T3+step  |  (T3+step = Datenende)</code></pre>
<p><strong>Fairer Baseline-Vergleich (origin-Protokoll):</strong> Jede Zeile
hat einen eigenen Informationsstand <code>cutoff = target &minus; h</code>.
Default "rolling": die Naive prognostiziert <code>y[cutoff]</code>, die
Seasonal-Naive den letzten beobachteten Wert mit gleichem Kalendermonat -
jeder bekommt exakt dieselben Informationen wie das LGBM zu derselben Zeile.
"fixed" (optional, M4-Klassik): alle Modelle prognostizieren vom Fold-Stichtag.
Metriken werden auf der gemeinsamen Nicht-NaN-Unterlage aller Modelle
berechnet; Spalte <code>n</code> berichtet die Unterlagegroesse.</p>
<p><strong>Metriken:</strong></p>
<ul>
<li><strong>MAE</strong> = Mittel |y&minus;&#375;| - robust, prim&auml;re Referenz</li>
<li><strong>RMSE</strong> = sqrt(Mittel (y&minus;&#375;)&sup2;) - bestraft Ausrei&szlig;er</li>
<li><strong>sMAPE</strong> = 200 % &times; Mittel |y&minus;&#375;| / (|y|+|&#375;|) - relativ, gr&ouml;&szlig;enunabh&auml;ngig</li>
<li><strong>Directional Accuracy</strong>: Anteil korrekter Richtungen
sign(&#375;&minus;y<sub>ref</sub>) = sign(y&minus;y<sub>ref</sub>) mit
y<sub>ref</sub> = letzter beobachteter Wert zum Forecast-Origin; Zeilen ohne
definierte Richtung - Bewegung 0 in Wahrheit <em>oder</em> in Vorhersage -
werden ausgeschlossen (deshalb ist die rolling-Naive richtungs-unscharf:
ihre Vorhersage IST der Referenzwert)</li>
<li><strong>MASE</strong> (nur E5): MAE skaliert mit dem In-Sample-Fehler der
1-Schritt-saisonalen Naive (m=12) derselben Serie; &lt; 1 hie&szlig;e besser
als diese Referenz</li>
</ul>
</div>""")

    # ------------------------------------------------------------- galerie
    parts.append("""
<h2 id="gallery">Zeitreihen-Galerie: Verl&auml;ufe mit Prognosen</h2>
<p>Metriken komprimieren viel - hier dasselbe bildlich. Alle Modelle
trainieren ausschlie&szlig;lich auf Daten vor dem gestrichelten Stichtag;
die schwarze dicke Linie ist die eingetretene Wahrheit.</p>
""")
    parts.append(
        fig(
            "e7_m4_examples",
            "M4 Beispiele",
            "Sechs reale M4-Monatsserien, 18-Monats-Prognose. Typische Muster: "
            "Seasonal-Naive (orange) wiederholt das Vorjahr exakt - auf trendenden "
            "Serien bleibt sie strukturell zur&uuml;ck; das Level-LGBM (grau) folgt "
            "dem Niveau, kann aber Trendkr&uuml;mmung nicht vorwegnehmen; direkt "
            "Log-Diff (t&uuml;rkis) extrapoliert Wachstumsraten.",
        )
    )
    parts.append("""
<div class="card finding">
<strong>Was man hier sieht.</strong> Die drei Prognose-Charaktere lassen sich
am Verlauf ablesen: Seasonal-Naive = Kopie des Vorjahres; Level-LGBM =
gegl&auml;ttete Fortschreibung, die bei Kurven zu flach wird; Log-Diff =
Wachstumsrate-Extrapolation, die bei Trendwechseln (siehe E6/Trendumkehr)
in die alte Richtung weiterl&auml;uft, solange nichts Neues im Training war.
</div>""")
    parts.append(
        fig(
            "e7_regime_examples",
            "Regime- x Saisonalitaets-Raster",
            "Raster aus E6-Trend-Regime (Zeilen) und Saisonstaerke (Spalten: ohne / "
            "schwach / stark), je Zelle eine Beispielserie mit allen vier Varianten. "
            "Ohne Saison - der Normalfall in Unternehmensdaten - bleibt "
            "Seasonal-Naive ohne Anker (kopiert nur Rauschen); die LGBM-Varianten "
            "unterscheiden sich vor allem &uuml;ber den Trend. In <em>trendumkehr</em> "
            "f&auml;ngt sich nur der rekursive Rollout, weil seine Eingaben mit der "
            "realen Abw&auml;rtsentwicklung aktualisiert werden.",
        )
    )

    # ------------------------------------------------------------------ e1
    if e1:
        parts.append("""
<h2 id="e1">E1 &middot; Szenario-Raster: wann gewinnt LGBM?</h2>
<p>F&uuml;nf kontrollierte Datenerzeugungsprozesse (Trend &times; Saisonalit&auml;t &times;
Rauschen) gegen die Baselines Naive und Seasonal-Naive. Je Szenario ein
eigenes Panel; identischer Backtest (3 Folds &times; 18 Monate).</p>
""")
        parts.append(
            setup_box(
                **{
                    "Serien": "50 je Szenario",
                    "Länge": "132 Monate",
                    "Horizonte": "1/6/12/18",
                    "Folds": "3",
                    "Boosting": "300 Runden",
                }
            )
        )
        parts.append(
            fig(
                "e1_scenario_grid",
                "MAE je Szenario und Modell",
                "Jedes Panel zeigt MAE &uuml;ber den Horizont f&uuml;r die drei Modelle. "
                "Je gr&ouml;&szlig;er der Abstand der blauen (LGBM) zur orangen Linie "
                "(Seasonal-Naive), desto gr&ouml;&szlig;er der LGBM-Vorteil.",
            )
        )
        parts.append("<h3>MAE-Ratio LGBM / Seasonal-Naive (&lt; 1 = LGBM besser)</h3>")
        parts.append(ratio_table(e1["lgbm_over_snaive_mae_ratio"]))
        parts.append("<h3>MAE im Detail (alle Szenarien)</h3>")
        flat = {}
        for scenario, mm in e1["metrics"].items():
            for model, hm in mm.items():
                flat[f"{scenario} · {model}"] = hm
        parts.append(metrics_table(flat, values=("mae",), digits=1))
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Sauber saisonal + station&auml;r:</strong> Seasonal-Naive ist nahezu
optimal (Ratio &asymp; 1). Die wahre Funktion ist zu einfach - ein Baummodell
kann hier nichts hinzuf&uuml;gen, au&szlig;er Rauschen zu lernen.</li>
<li><strong>Rauschen dreht das Blatt:</strong> LGBM mittelt &uuml;ber 50 Serien und
gewinnt 7-15&nbsp;% (Ratio 0.85-0.94) - Panel-Learning als Rauschfilter.</li>
<li><strong>Trend ist der gr&ouml;&szlig;te Hebel:</strong> Bei exponentiellem Wachstum
bleibt Seasonal-Naive 12 Monate zur&uuml;ck; LGBM erreicht nur 17-29&nbsp;% deren
Fehler, weil Year-over-Year-Differenzen das Wachstum extrapolieren.</li>
<li><strong>Auch ohne Saisonalit&auml;t gewinnt LGBM</strong> auf strukturllosen
Serien (Ratio 0.76-0.80): die Serien-ID als kategoriales Feature lernt
unterschiedliche Niveaus.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e2
    if e2:
        s = e2["scenarios"]
        parts.append("""
<h2 id="e2">E2 &middot; Datenaufbereitung auf stark trendenden Daten</h2>
<p>Exponentieller Trend (monatliche Wachstumsrate 1.5-3.5&nbsp;%), saisonale
Komponente, moderates Rauschen. Vier Zieltransformationen, sonst identisches
Setup und identische Folds. Rekonstruktion der Levels erfolgt leakage-frei
aus beobachteten Ankern (bei Saisondifferenzen: y[t+h&minus;12], nur Horizonte
&le; 12).</p>
""")
        parts.append(
            setup_box(
                **{
                    "Serien": "60",
                    "Länge": "132 Monate",
                    "Horizonte": "1/6/12",
                    "Folds": "3",
                    "Wachstum": "1.5-3.5 %/Monat",
                    "Saison": "15-35 abs.",
                }
            )
        )
        parts.append(
            fig(
                "e2_data_prep",
                "MAE und Richtungsguete je Transformation",
                "Links MAE (je niedriger desto besser), rechts Directional Accuracy "
                "(0.5 = M&uuml;nzwurf, 1.0 = perfekte Richtung). Achte auf den "
                "Niveaunterschied zwischen <em>levels</em> und den drei "
                "transformierten Varianten.",
            )
        )
        parts.append("<h3>Vollst&auml;ndige Metriken je Transformation</h3>")
        parts.append(
            metrics_table(
                s, values=("mae", "rmse", "dir_acc"), digits=2, lower_is_better={"dir_acc": False}
            )
        )
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Rohe Levels versagen kategorial</strong> (MAE im vierstelligen
Bereich, RMSE nochmal ein Vielfaches dar&uuml;ber): Baummodelle
interpolieren nur innerhalb des trainierten Wertebereichs. Ein Wert, der
historisch nie vorkam (n&auml;chste Stufe des Exponentialtrends), kann nicht
ausgesprochen werden.</li>
<li><strong>Additive Saisondifferenzen</strong> (y<sub>t</sub>&minus;y<sub>t&#8202;&minus;&#8202;12</sub>)
reduzieren den Fehler um Faktor ~3.4 - die Serie wird station&auml;rer, aber
die Amplitude der Differenzen w&auml;chst mit dem Level weiter mit.</li>
<li><strong>Log-Saisondifferenzen gewinnen um Gr&ouml;&szlig;enordnungen</strong>
(MAE ~10 statt ~1190 bei h=12): Im Lograum ist die Serie homoskedastisch,
Extrapolation wird zur Sch&auml;tzung einer stabilen Wachstumsrate.
Directional Accuracy nahe 1.0 auf allen Horizonten.</li>
<li>Reines <code>log</code> (ohne Differenzen) hilft deutlich, bleibt aber
hinter den Differenz-Varianten zur&uuml;ck - der Level-Charakter bleibt
teilweise erhalten.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e3
    if e3:
        parts.append("""
<h2 id="e3">E3 &middot; Feature-Ablation: was beschreibt die Serie?</h2>
<p>DGP mit bekanntem exogenem Treiber x (AR(1), &phi;=0.7, kausal mit 1 Monat
Verz&ouml;gerung): y = level + &beta;&middot;x<sub>t&#8202;&minus;&#8202;1</sub> + Saison +
Trend + Rauschen. Sechs Feature-Sets, identische Folds und Boosting.</p>
""")
        parts.append(
            setup_box(
                **{
                    "Serien": "60",
                    "Länge": "132 Monate",
                    "Horizonte": "1/6/12/18",
                    "Folds": "3",
                    "Treiber": "AR(1), φ=0.7, β∈[1.8,2.6]",
                }
            )
        )
        parts.append(
            fig(
                "e3_feature_ablation",
                "Feature-Ablation MAE und Importance",
                "Links: MAE-Kurven der sechs Feature-Sets - je tiefer, desto besser; "
                "die Reihenfolge der Farben folgt der Ablationskette. Rechts: "
                "Gain-Anteile je Feature-Familie des vollst&auml;ndigen Modells bei h=12.",
            )
        )
        parts.append("<h3>MAE je Feature-Set</h3>")
        parts.append(metrics_table(e3["metrics"], values=("mae", "rmse"), digits=2))
        sh = e3["importance_share_h12"]
        share_rows = "".join(
            f"<tr><td>{k}</td><td>{fmt(100 * v, 1)}&nbsp;%</td></tr>"
            for k, v in sorted(sh.items(), key=lambda kv: -kv[1])
        )
        parts.append(
            "<h3>Gain-Anteile je Familie (h=12, Modell inkl. Treiber)</h3>"
            f"<table><thead><tr><td>Familie</td><th>Anteil</th></tr></thead>"
            f"<tbody>{share_rows}</tbody></table>"
        )
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Nur Target-Lags ist die schw&auml;chste Konfiguration.</strong>
Rolling-Statistiken bringen den gr&ouml;&szlig;ten Einzelsprung (Gl&auml;ttung +
Niveau-Information), Kalenderfeatures helfen vor allem am langen Horizont
(Jahreszeit-Einordnung ohne Lag-12-Anchor).</li>
<li><strong>Der kausale Treiber x hilft massiv bei h=1</strong> (MAE 8.9 vs
11.8, ~25&nbsp;% besser) und verliert mit wachsender Distanz: AR(1) mit
&phi;=0.7 hat nach 6-12 Monaten fast keine Erinnerung - der Treiber ist zwar
kausal relevant, aber selbst kaum prognostizierbar. <em>Kausale Relevanz
&ne; Prognosenutzen</em>, es sei denn der Treiber ist persistent oder
zuk&uuml;nftig bekannt (Pl&auml;ne!).</li>
<li><strong>Gain-Importance verteilt sich trotz bekannter Kausalit&auml;t auf
Target-Lags und Rolling-Stats:</strong> korrelierte, redundante Features
teilen sich die Attribution. Importance-Bilder sind Erkl&auml;rungs-Anker,
aber keine Kausalit&auml;tsbeweise.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e4
    if e4:
        parts.append("""
<h2 id="e4">E4 &middot; Kausale Plausibilit&auml;t: Vorhersage ist nicht Erkl&auml;rung</h2>
<p>Synthetische Welt mit persistentem Budget-Treiber x (OU-Prozess um Level
45, &phi;=0.9). y reagiert kausal mit einem Monat Verz&ouml;gerung
(&beta;&isin;[2,3]). Nach 132 Trainingsmonaten wird x per do-Operator auf
35&nbsp;% gesenkt - die Counterfactual-Welt ist bekannt, weil das DGP
synthetisch ist. Drei Modelle, trainiert ausschlie&szlig;lich auf
Pr&auml;-Interventionsdaten:</p>
<ul>
<li><strong>lag_only</strong>: Target-Lags/Rolling/Kalender (kein x)</li>
<li><strong>with_x</strong>: + aktueller x-Stand (sieht die Senkung nicht)</li>
<li><strong>with_x_plan</strong>: + geplanter Pfad von x als
Szenario-Feature (Wert bei Ziel&minus;1 Monat) - realistisch, weil
Budgetpl&auml;ne zum Forecast-Zeitpunkt bekannt sind</li>
</ul>
""")
        parts.append(
            setup_box(
                **{
                    "Serien": "40",
                    "Länge": "150 Monate",
                    "Intervention": "Monat 132, x → 35 %",
                    "Origin": "T=132",
                    "Fenster": "18 Monate",
                    "Boosting": "300 Runden",
                }
            )
        )
        parts.append(
            fig(
                "e4_causal_intervention",
                "Interventionsexperiment",
                "Links (Beispielserie S00): schwarze Linie = Wahrheit nach "
                "Budget-Senkung, grau gestrichelt = Welt ohne Senkung. Die rote "
                "Prognose (lag_only) bleibt oben im alten Regime, t&uuml;rkis "
                "(with_x_plan) folgt dem Einbruch. Mitte: MAE je Horizont im "
                "Interventionsfenster. Rechts: Gain-Anteile beider Modelle.",
            )
        )
        parts.append("<h3>Regime-Vergleich vor der Intervention (honest Backtest)</h3>")
        parts.append(metrics_table(e4["regime_backtest"], values=("mae",), digits=2))
        parts.append("<h3>Prognosen im Interventionsfenster (18 Monate ab Stichtag)</h3>")
        iv_rows = "".join(
            f"<tr><td>{r['model']}</td><td>{fmt(r['mae'], 1)}</td>"
            f"<td class='{'lo' if r['bias'] > 30 else ''}'>{fmt(r['bias'], 1)}</td>"
            f"<td>{fmt(100 * r['dir_acc'], 1)}&nbsp;%</td></tr>"
            for r in e4["intervention"]
        )
        parts.append(
            "<table><thead><tr><td>Modell</td><th>MAE</th><th>Bias</th><th>Dir.&nbsp;Acc</th></tr></thead>"
            f"<tbody>{iv_rows}</tbody></table>"
            "<p class='figread'>Bias = Mittel(&#375;&minus;y): positiv hei&szlig;t das "
            "Modell &uuml;bersch&auml;tzt systematisch - es hat den Einbruch nicht "
            "verstanden. Dir. Acc relativ zum letzten beobachteten Wert.</p>"
        )
        parts.append(f"""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Im normalen Regime sieht alles gut aus:</strong> Der Backtest
unterscheidet die Modelle kaum - <code>with_x</code> ist sogar fl&auml;chendeckend
leicht besser. Die Prognosequalit&auml;t allein verr&auml;t nicht, welches Modell
den Kausalzusammenhang verstanden hat.</li>
<li><strong>Nach dem Eingriff extrapolieren lag_only und with_x das alte
Regime:</strong> Bias +66, Richtungstreffen auf M&uuml;nzwurf-Niveau
(&asymp;49&nbsp;%). Die Lags kodieren die korrelierte Vergangenheit, nicht die
Ursache.</li>
<li><strong>Nur das Szenario-Modell reagiert</strong> (Dir.&nbsp;Acc
{fmt(100 * (e4_dir or 0), 0)}&nbsp;%), aber es erfasst nur ~45&nbsp;% der wahren
Effektst&auml;rke: die redundanten Rolling-Features, die im Regime so
hilfreich waren, binden Gewicht und dilutieren die Antwort. <em>Wer
Interventionsf&auml;higkeit will, muss Feature-Redundanz reduzieren oder
explizite Struktur vorgeben.</em></li>
<li><strong>Gain-Importance als Erkl&auml;rungs-Anker:</strong> ohne x liegt die
gesamte Masse auf Lags/Rolling/Serien-ID; mit x bekommt der Treiber einen
sichtbaren Anteil - die Importance zeigt, <em>wor&uuml;ber</em> das Modell
spricht, nicht ob es kausal richtig liegt.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e5
    if e5:
        parts.append(f"""
<h2 id="e5">E5 &middot; Realer Benchmark: M4-Monatsdaten</h2>
<p>{e5["n_series"]} zuf&auml;llig gezogene M4-Monatsserien (Wettbewerb: Monats-
daten, Horizont 18), 2 Folds &times; 18 Monate Testfenster. Protokoll: Folds
sind je Serie an deren eigenem Ende verankert (M4-typische ungleiche
Serienl&auml;ngen), Baselines prognostizieren rolling-origin aus demselben
Informationsstand wie LGBM, und alle Metriken werden auf der gemeinsamen
Nicht-NaN-Unterlage aller Modelle berechnet (Spalte <code>n</code>).</p>
""")
        parts.append(
            setup_box(
                **{
                    "Serien": str(e5["n_series"]),
                    "Folds": "2 × 18 Monate",
                    "Horizonte": "1/6/12/18",
                    "Boosting": "400 Runden",
                }
            )
        )
        parts.append(
            fig(
                "e5_m4_benchmark",
                "M4 Benchmark",
                "Links MAE, Mitte sMAPE (beide je Horizont), rechts MASE als "
                "Gesamtkennzahl. Auf echten Daten sind die Niveaus gro&szlig; und "
                "serienheterogen - die <em>Relativordnung</em> der Modelle ist die "
                "Aussage, nicht der absolute Wert.",
            )
        )
        parts.append("<h3>Metriken je Horizont</h3>")
        parts.append(metrics_table(e5["metrics"], values=("mae", "smape"), digits=2))
        mo = e5["mase_overall"]
        mase_rows = "".join(
            f"<tr><td>{k}</td><td{' class=hl' if v == min(mo.values()) else ''}>{fmt(float(v), 3)}</td></tr>"
            for k, v in sorted(mo.items(), key=lambda kv: kv[1])
        )
        parts.append(
            "<h3>MASE (gegen In-Sample-Seasonal-Naive skaliert)</h3>"
            "<div class='card'><p>MASE &lt; 1 hie&szlig;e: besser als die saisonale"
            " Naive auf dem eigenen Historieniveau. <strong>Alle drei Modelle"
            " liegen dar&uuml;ber</strong> - 18-Monats-Forecasts auf M4-Monatsdaten"
            " schlagen die 1-Schritt-In-Sample-Referenz nicht; die <em>Reihenfolge</em>"
            " LGBM vor Naive vor Seasonal-Naive ist die Aussage. Die MAE-/sMAPE-Charts"
            " oben bleiben die prim&auml;re Referenz.</p></div>"
            f"<table><thead><tr><td>Modell</td><th>MASE</th></tr></thead>"
            f"<tbody>{mase_rows}</tbody></table>"
        )
        imp = e5.get("importance_lgbm_h18_top10") or {}
        if imp:
            imp_rows = "".join(
                f"<tr><td>{k}</td><td>{fmt(100 * v, 1)}&nbsp;%</td></tr>" for k, v in imp.items()
            )
            parts.append(
                "<h3>LGBM-Gain-Anteile bei h=18 (Top 10)</h3>"
                "<p class='figread'>Auf echten Daten tragen Lags und Rolling-Stats "
                "die Hauptlast; Kalenderfeatures (month) sind relevant, "
                "Saisondifferenzen (diff_12) messbar.</p>"
                f"<table><thead><tr><td>Feature</td><th>Anteil</th></tr></thead>"
                f"<tbody>{imp_rows}</tbody></table>"
            )
        parts.append("""
<div class="card finding">
<strong>Befunde (korrigiertes Protokoll - die fr&uuml;heren
"41&nbsp;% besser"-Zahlen waren ein Artefakt eingefrorener Baselines).</strong>
<ul>
<li><strong>Kurzer Horizont: Gleichstand mit der Naive</strong> (h=1: MAE
284 vs 284). Am 1-Schritt-Fenster einer echten Serie ist ein globales
Baummodell nichts Besseres als "letzter Wert" - mehr Struktur gibt es dort
nicht zu lernen.</li>
<li><strong>Mittlerer Horizont: LGBM vorn</strong> (h=6: MAE 434 vs 489/538,
~11&nbsp;% besser als die Naive); <strong>bei h=12 knapp dahinter</strong>
(534 vs 518), <strong>bei h=18 wieder vorn</strong> (658 vs 697/763).
LGBM gewinnt 3 von 4 Buckets, aber keineswegs &uuml;berall.</li>
<li><strong>Directional Accuracy:</strong> wo definiert, trifft LGBM die
Richtung am h&auml;ufigsten (~0.56-0.63 vs ~0.55-0.58 Seasonal-Naive). F&uuml;r
die rolling-Naive ist Richtung per Konstruktion undefiniert
(pred = letzter Wert, siehe Methoden-Box) und wird als n.v. berichtet.</li>
<li><strong>Ehrliche Einordnung:</strong> E5 testet die schw&auml;chste
Konfiguration (Levels + Default-Features). Der E2/E6-Hebel
(Log-Saisondifferenzen) bleibt die offene Verbesserung auf echten Daten -
siehe E11.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e6
    if e6:
        mm = e6["metrics_on_levels"]
        parts.append("""
<h2 id="e6">E6 &middot; Level vs. Log-Differenzen &uuml;ber f&uuml;nf Trend-Regime</h2>
<p>Die Kernfragen: <em>Sind direkte Level-Forecasts besser als
Log-Differenz-Forecasts? Was kostet die Rekursivit&auml;t der Differenzen?
Und wie &auml;ndert sich die Antwort mit Form und Richtung des Trends?</em>
F&uuml;nf Regime, jeweils identischer Vier-Vergleich (60 Serien &times; 144 Monate,
Horizonte 1-18 in Monatsaufl&ouml;sung):</p>
<ul>
<li><strong>kein_trend</strong>: station&auml;r + Saison</li>
<li><strong>leicht_trendend</strong>: exponentiell, 0.2-0.5&nbsp;%/Monat (&asymp;2.4-6&nbsp;%/Jahr)</li>
<li><strong>linear_trendend</strong>: additiv +2..6 Einheiten/Monat</li>
<li><strong>stark_trendend</strong>: exponentiell, 1.2-3&nbsp;%/Monat</li>
<li><strong>trendumkehr</strong>: +1.5-2.5&nbsp;%/Monat bis Monat 96, danach
&minus;60&nbsp;% der Rate (Strukturbruch)</li>
</ul>
<p>Varianten: <strong>direct_level</strong> (LGBM pro Horizont auf Levels),
<strong>direct_logdiff</strong> (Label = h-Schritt-Aenderung
<code>log y[t+h] &minus; log y[t]</code>, Rekonstruktion aus beobachtetem Anker,
keine Rekursion), <strong>recursive_logdiff</strong> (ein 1-Schritt-Modell,
Rollout mit eigenen Prognosen als Lags), <strong>seasonal_naive</strong>.</p>
""")
        parts.append(
            setup_box(
                **{
                    "Serien": "60 je Regime",
                    "Länge": "144 Monate",
                    "Folds": "3 × 18 Monate",
                    "Boosting": "300 Runden",
                    "Level-Bereich": "80-400 (log-sicher)",
                }
            )
        )
        parts.append(
            fig(
                "e6_levels_vs_logdiff",
                "Level vs Log-Diff ueber Regime",
                "Ein Panel je Regime, MAE auf Levels, logarithmische Y-Achse. "
                "Beobachte, wie sich die Ordnung der Kurven zwischen "
                "<em>stark_trendend</em> und <em>trendumkehr</em> umkehrt.",
            )
        )

        def e6_rows(regimes: dict) -> str:
            out = []
            for regime, models in regimes.items():
                out.append(f'<tr class="group"><td colspan="5">{regime}</td></tr>')
                for model in (
                    "direct_level",
                    "seasonal_naive",
                    "recursive_logdiff",
                    "direct_logdiff",
                ):
                    hm = models.get(model, {})
                    cells = "".join(
                        f"<td>{fmt(hm.get(h, {}).get('mae'), 1)}</td>"
                        for h in ("1", "6", "12", "18")
                    )
                    out.append(f"<tr><td>{model}</td>{cells}</tr>")
            return "".join(out)

        parts.append("<h3>MAE je Regime und Variante</h3>")
        parts.append(
            "<table><thead><tr><td>Regime / Variante</td><th>h=1</th>"
            "<th>h=6</th><th>h=12</th><th>h=18</th></tr></thead>"
            f"<tbody>{e6_rows(mm)}</tbody></table>"
        )
        parts.append("""
<div class="card finding">
<strong>Befunde je Regime.</strong>
<ul>
<li><strong>kein Trend:</strong> Alles gleichwertig (Levels 4.1 vs direkt
Log-Diff 4.8 bei h=18). Ohne Trend kein Extrapolationsproblem - Transformation
bring nichts, Seasonal-Naive ist hier sogar okay (4.3).</li>
<li><strong>leichter Trend:</strong> Bei h=18 praktisch unentschieden
(Levels 6.3 vs Log-Diff 6.8). Der Wertebereich verschiebt sich in 18 Monaten
nur wenig - das Level-Handikap existiert noch kaum. Am kurzen Horizont ist
direkt Log-Diff klar vorne (4.1 vs 5.7 bei h=1).</li>
<li><strong>linearer Trend:</strong> Levels halten sich bemerkenswert gut
(11.5 vs 11.5 bei h=18): ein additiver Trend w&auml;chst langsam relativ zum
Wertebereich, B&auml;ume interpolieren den Gro&szlig;teil &uuml;ber die
Year-over-Year-Lags.</li>
<li><strong>starker Trend:</strong> Das klassische Bild in Reinform -
Levels 305 vs direkt Log-Diff 13.2 bei h=18 (<strong>23&times;</strong>);
rekursiv 176 (<strong>13&times; Rekursions-Strafe</strong>, fast linearer
Zuwachs ~9.2/Monat).</li>
<li><strong>Trendumkehr:</strong> Die Rangfolge dreht sich! Rekursiv gewinnt
(64.6 vs 196.5 direkt Log-Diff bzw. 186.9 Levels bei h=18). Grund: Die
h-Schritt-Labels der direkten Modelle stammen aus dem alten Wachstumsregime;
der rekursive Rollout sieht die Kehre nach wenigen Schritten in seiner
Historie und passt an. <em>Strukturbr&uuml;che bestrafen die eingefrorene
Direkt-Formulierung h&auml;rter.</em></li>
</ul>
</div>
<div class="card warn">
<strong>Praxis-Konsequenz.</strong> Auf stabilen Trends gilt: Log-Transformation,
h-Schritt-Log-Differenz als Label, ein Booster pro Horizont (E2+E6). Nach
Strukturbr&uuml;chen kehrt sich der Vorteil um - hier helfen kurze
Retraining-Zyklen, adaptive rekursive Rollouts oder explizite
Bruch-Erkennung. Die Wahl des Prognose-Setups ist also eine Frage der
Regime-Stabilit&auml;t, nicht eine Geschmacksfrage.
</div>""")

    # ------------------------------------------------------------------ e8
    if e8:
        mm8 = e8["metrics_on_levels"]
        parts.append("""
<h2 id="e8">E8 &middot; Alles zusammen: Trend, Treiber und Ans&auml;tze konfrontiert</h2>
<p>E1-E7 haben Effekte <em>isoliert</em> - je Studie ein DGP f&uuml;r eine
Frage. Hier kombiniert ein DGP alle Eigenschaften gleichzeitig: moderater
Exponentialtrend (<strong>+5-11&nbsp;%/Jahr</strong>), moderate Saison,
persistenter exogener Treiber <code>x</code> (AR(1) um 45, Wirkung auf den
Folgemonat) und AR-Rauschen. Daraus ein 2&times;2-Faktor
(Formulierung &times; Treiber-Info) plus Referenz:</p>
<ul>
<li><strong>levels / levels_x</strong>: direktes LGBM auf Levels, ohne/mit <code>x</code></li>
<li><strong>logdiff / logdiff_x</strong>: direkt auf h-Schritt-Log-Aenderungen,
ohne/mit <code>x</code></li>
<li><strong>seasonal_naive</strong> als Referenz</li>
</ul>
""")
        parts.append(
            setup_box(
                **{
                    "Serien": "60",
                    "Länge": "144 Monate",
                    "Horizonte": "1/3/6/12/18",
                    "Folds": "3 × 18 Monate",
                    "Trend": "+0.4-0.9 %/Monat",
                    "Saison": "Amplitude 8-18",
                    "Treiber x": "AR(1), φ=0.9, β∈[1.5,2.5]",
                    "Boosting": "300 Runden",
                }
            )
        )
        parts.append(
            fig(
                "e8_combined",
                "Alles kombiniert",
                "Links MAE auf Levels (log-Skala): Levels starten schon bei h=1 hoch "
                "(das Niveau-Problem), Log-Diffs fixen das, und der Treiber addiert "
                "auf beiden Formulierungen Schub. Rechts der Gain-Anteil von x: am "
                "kurzen Ende ~44&nbsp;%, am langen Ende ~19&nbsp;%.",
            )
        )
        parts.append("<h3>Metriken je Variante (alle Horizonte)</h3>")
        parts.append(
            metrics_table(
                mm8, values=("mae", "dir_acc"), digits=2, lower_is_better={"dir_acc": False}
            )
        )
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Die Vorteile addieren sich:</strong> MAE bei h=18 l&auml;uft
20.8 (Levels) &rarr; 16.8 (Log-Diff) &rarr; <strong>13.5 (Log-Diff + x)</strong>.
Kein Einzeltrick gewinnt - die Kombination schl&auml;gt jede Teill&ouml;sung.</li>
<li><strong>Der Treiber entfaltet Wirkung erst oberhalb der richtigen
Formulierung:</strong> Auf Levels bringt x nur ~8&nbsp;% (20.8&rarr;19.1),
auf Log-Diffs ~20&nbsp;% (16.8&rarr;13.5). Solange das Modell mit dem Niveau
k&auml;mpft, frisst das Trend-Problem den Nutzen des Fuehrungssignals auf.</li>
<li><strong>x wirkt vor allem kurzfristig:</strong> Gain-Anteil ~44&nbsp;% bei
h=1 gegen ~19&nbsp;% bei h=18. Fuer den Folgemonat ist der aktuelle
Treiberstand ein starkes Signal; fuer 18 Monate dominieren Lags und Trend.</li>
<li><strong>Directional Accuracy in derselben Rangfolge:</strong> h=1:
82&nbsp;% (logdiff_x) vs 61&nbsp;% (levels).</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e9
    if e9:
        parts.append("""
<h2 id="e9">E9 &middot; Hyperparameter-Tuning: der kleine Hebel</h2>
<p>LightGBM kommt mit vern&uuml;nftigen Defaults. Wie viel gewinnt man, wenn
Optuna (TPE, 40 Trials) die Kern-Hyperparameter <em>pro Szenario</em>
optimiert? Zeitlich sauber getrennt: die Suche sieht nur Fold 1, bewertet
wird auf dem sp&auml;teren Fold 2.</p>
""")
        rows = ""
        label = {
            "kein_trend": "Kein Trend + Saison",
            "stark_trendend": "Starker Exponentialtrend",
            "trendumkehr": "Strukturbruch im Trend",
            "exog_treiber": "Trend + Treiber x",
            "m4_real": "M4, 150 echte Serien",
        }
        for k, v in e9["scenarios"].items():
            imp = v["improvement_pct"]
            cls = ' class="hl"' if imp > 0 else ' class="lo"'
            rows += (
                f"<tr><td>{label.get(k, k)}</td>"
                f"<td>{v['mae_holdout_default']:.2f}</td>"
                f"<td>{v['mae_holdout_tuned']:.2f}</td>"
                f"<td{cls}>{imp:+.1f}&nbsp;%</td></tr>"
            )
        parts.append(
            "<table><thead><tr><td>Szenario</td><th>MAE Default</th>"
            "<th>MAE Tuned</th><th>&Delta;</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Tuning ist ein Nebenhebel:</strong> Der beste Gewinn &uuml;ber
f&uuml;nf Szenarien liegt bei +5,8&nbsp;% (M4). Formulierung (E2: Faktor ~120)
und Features (E3/E8: ~20&nbsp;%) sind Gr&ouml;&szlig;enordnungen st&auml;rker.</li>
<li><strong>Tuning kann aktiv schaden:</strong> Beim Strukturbruch kostet das
auf Ruhe getrimmte Setup &minus;16&nbsp;% - Hyperparameter sind Regime-Annahmen.
Wer auf dem letzten Bruch tunet, optimiert die Vergangenheit fest.</li>
<li><strong>Praxis-Fazit:</strong> Defaults halten, Energie in Label-
Formulierung und Feature-Auswahl stecken - erst wenn beide stehen, lohnt
sich Feinschliff mit sauberem Holdout.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e10
    if e10:
        s = e10["setup"]
        b = e10["budget"]
        rev = e10["revision"]["family_share_of_abs_revision"]
        prof1 = e10["profiles"]["1"]
        prof18 = e10["profiles"]["18"]
        rec = e10["recovery"]
        parts.append("""
<h2 id="e10">E10 &middot; Was treibt die Prognose? TreeSHAP gegen das wahre DGP</h2>
<div class="card">
<p><strong>Ansatz.</strong> Importance-Zahlen (Gain, Splits) sagen nicht,
was ein Modell <em>wirklich</em> nutzt. Deshalb hier der sch&auml;rfste Test,
den Synthetik erlaubt: Wir kennen das DGP (60&nbsp;Serien &times;&nbsp;168&nbsp;Monate,
kausaler AR(1)-Treiber <code>x</code> mit Serie-spezifischem
&beta;&nbsp;&isin;&nbsp;[1.8,&nbsp;2.6], Wirkung auf den Folgemonat). Ein globales
direktes LGBM wird auf Zielen bis Monat 114 trainiert und <em>eingefroren</em>;
die TreeSHAP-Werte nativ via <code>pred_contrib=True</code> werden nur auf den
ungesehenen letzten 54&nbsp;Monaten berechnet - produktionstreu und ohne
neue Abh&auml;ngigkeit. TreeSHAP ist exakt additiv:
Prognose = Basiswert + Summe aller Beitr&auml;ge (max. Abweichung hier
6.5e-13).</p>
</div>
""")
        parts.append(
            fig(
                "e10_family_budget",
                "Erklaerungs-Budgets",
                "<b>Lesehilfe:</b> Je Familie drei Balken - was mean|SHAP| als "
                "Erkl\u00e4rung verteilt (blau), was Gain-Importance nennt (orange), "
                "und was laut DGP tats\u00e4chlich wirkt (gr\u00fcn). Lags und Rolling "
                "sind kausal leer, schlucken aber ~71 % des SHAP-Budgets - sie sind "
                "die besten <i>Proxies</i>. Der wahre Treiber tr\u00e4gt 24 % des "
                "Signals, bekommt aber nur 3.5 %.",
            )
        )
        parts.append(f"""
<div class="card finding">
<strong>A) Budgets.</strong> Gepoolt \u00fcber alle Horizonte: Rolling {100 * b["shap"].get("Rolling-Stats", 0):.0f}&nbsp;%
+ Lags {100 * b["shap"].get("Target-Lags", 0):.0f}&nbsp;% dominieren das SHAP-Budget;
Treiber&nbsp;x erh&auml;lt {100 * b["shap"].get("Treiber x", 0):.1f}&nbsp;% (Gain: {100 * b["gain"].get("Treiber x", 0):.1f}&nbsp;%),
obwohl er {100 * b["truth"].get("Treiber x", 0):.0f}&nbsp;% des wahren Signals tr&auml;gt.
<em>Predictive Budget &ne; kausales Budget</em> - quantifizierte Version der E4-Lektion.
</div>""")
        parts.append(
            fig(
                "e10_recovery",
                "Koeffizienten-Recovery",
                "<b>Lesehilfe:</b> Links die Steigung von SHAP(x) gegen x je Horizont. "
                "Bei h=1 ist x exakt der kausale Eingang - die Steigung trifft \u03b2\u0304 "
                "(2.10 vs 2.21). Nach rechts d\u00fcrfte sie nur wie \u03b2\u00b7\u03c6^(h-1) "
                "(rot) zerfallen, weil das Modell x_{t+h-1} nie sehen kann. Tats\u00e4chlich "
                "verstummt der Treiber ab h\u22484: Boosting mit Regularisierung l\u00e4sst "
                "schwache, von Lags abgesattelte Signale fallen. Rechts: Recovery je "
                "Serie bei h=1 - Richtung stimmt, Range komprimiert (r=0.46, MAE=0.19).",
            )
        )
        parts.append(f"""
<div class="card finding">
<strong>B) Recovery.</strong> Bei h=1 rekonstruiert SHAP den kausalen Koeffizienten
(Steigung {rec["slope_decay"][0]["slope"]:.2f} vs. Referenz {rec["slope_decay"][0]["ref"]:.2f};
je Serie r={rec["beta_corr_h1"]:.2f}, MAE={rec["beta_mae_h1"]:.2f}). Die gemessene
Zerfallskurve f\u00e4llt aber deutlich schneller als \u03b2\u00b7\u03c6^(h-1): Das
<em>gefittete</em> Modell nutzt den Treiber k\u00fcrzer, als es die Bayes-Optimalit\u00e4t
erlauben w\u00fcrde - konsistent mit E8 (Gain-Anteil von x: 44&nbsp;% bei h=1,
19&nbsp;% bei h=18).
</div>""")
        parts.append(
            fig(
                "e10_horizon_profile",
                "Horizont-Profile",
                "<b>Lesehilfe:</b> Gestapelte Familienanteile je Horizont. Kurzfristig "
                "(h=1) tr\u00e4gt der Treiber ~16 %, langfristig (h=18) nur noch ~2 % - "
                "dort \u00fcbernehmen Lags (47 %), Entit\u00e4ts- und Kalendermerkmale.",
            )
        )
        parts.append(f"""
<div class="card finding">
<strong>C) Horizont-Profile.</strong> Treiber-Anteil {100 * prof1.get("Treiber x", 0):.0f}&nbsp;% bei h=1
gegen {100 * prof18.get("Treiber x", 0):.0f}&nbsp;% bei h=18. Wer Treiber-Szenarien
rechnet, darf sie nur kurzfristig wirken lassen - oder muss Pfad-Features
(Wert am Zieltermin) nutzen wie in E4.
</div>""")
        parts.append(
            fig(
                "e10_revision",
                "Revisionen erklaeren",
                "<b>Lesehilfe:</b> Zwei Origins (Monat 105 vs 106) erkl\u00e4ren denselben "
                "Zielmonat; die SHAP-Differenz zerlegt die Forecast-\u00c4nderung. Rechts "
                "ein Monat, in dem sich der Treiber stark bewegte (\u0394x = -19): fast die "
                "ganze Revision l\u00e4uft \u00fcber Treiber- und Lag-Block. Ganz au\u00dfen "
                "der ruhige Treiber: dort stammt die Revision aus Rolling/Lags. Additivit\u00e4t "
                "ist exakt - die Differenz erkl\u00e4rt die Revision vollst\u00e4ndig.",
            )
        )
        parts.append(f"""
<div class="card finding">
<strong>D) Revisionen.</strong> Im Mittel entf\u00e4llt die gr\u00f6\u00dfte
Revisionsmasse auf Target-Lags ({100 * rev.get("Target-Lags", 0):.0f}&nbsp;%)
und Rolling ({100 * rev.get("Rolling-Stats", 0):.0f}&nbsp;%); der Treiber tr\u00e4gt
den Rest. Praxis-Nutzung: Forecasts gegen&uuml;ber Stakeholdern als
SHAP-Differenz zweier Origins erkl&auml;ren statt als Blackbox-Update.
</div>""")
    # ------------------------------------------------------------------ e11
    if e11:
        mo = e11["mase_overall"]
        best = min(mo, key=mo.get)
        parts.append("""
<h2 id="e11">E11 &middot; Beste Formulierung gegen klassische lokale Modelle (M4)</h2>
<p>Fixed-Origin-Blockprognosen wie im M4-Wettbewerb: jedes Modell
prognostiziert 18 Monate vom eigenen fold_end. Arme: LGBM Levels (E5-Referenz),
LGBM Log-Diff (E6-Gewinner), Ensemble beider, LGBM je Einzelserie
(Cross-Learning aus), AutoETS und Theta als klassische lokale Kontrollen.
Metriken auf gemeinsamer Unterlage.</p>
""")
        parts.append(
            fig(
                "e11_m4_best",
                "Best-Formulation Benchmark",
                "MAE ueber den Horizont. Gestrichelt: Baselines; durchgezogen: "
                "ML-Arme und klassische lokale Modelle.",
            )
        )
        mase_rows = "".join(
            f"<tr><td>{k}</td><td{' class=hl' if k == best else ''}>{fmt(float(v), 3)}</td></tr>"
            for k, v in sorted(mo.items(), key=lambda kv: kv[1])
        )
        parts.append(f"""
<h3>MASE je Arm</h3>
<table><thead><tr><td>Modell</td><th>MASE</th></tr></thead>
<tbody>{mase_rows}</tbody></table>
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Klassische lokale Modelle sind auf klassischem M4 hart:</strong>
AutoETS und Theta schlagen alle LGBM-Varianten (MASE&nbsp;&asymp;&nbsp;0.93,
als einzige unter 1). Das M4-Bild "statistische Verfahren vorn" gilt hier
weiterhin - anders als in M5 (Retail, viele aehnliche Serien).</li>
<li><strong>Cross-Learning ist real:</strong> globales Levels-LGBM (1.09)
schlaegt dasselbe Modell je Serie (1.34) um ~18&nbsp;% MAE - aber es hebt
das LGBM nicht ueber die lokalen Klassiker.</li>
<li><strong>Ensemble hilft:</strong> Mittel aus Levels+Log-Diff ist der beste
ML-Arm (1.03) - die zwei Formulierungen machen unterschiedliche Fehler.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e12
    if e12:
        rows = e12["rows"]
        cov_rows = "".join(
            f"<tr><td>{r['method']}</td><td>{int(r['horizon'])}</td>"
            f"<td>{fmt(100 * float(r['coverage']), 1)}&nbsp;%</td>"
            f"<td>{fmt(float(r['width']), 0)}</td></tr>"
            for r in rows
        )
        parts.append("""
<h2 id="e12">E12 &middot; Prognoseintervalle: Quantil-Regression vs. Conformal</h2>
<p>Beide Ansaetze im Log-Raum (Skalen-Heterogenitaet), ausgewertet auf der
Level-Skala: LightGBM-Quantile-Booster (alpha = 0.1/0.9) gegen
Split-Conformal um die Punktprognose (signierte Residuen-Quantile,
Kalibration auf den letzten 25&nbsp;% der Trainingsziele).</p>
""")
        parts.append(
            fig(
                "e12_intervals",
                "Prognoseintervalle",
                "Links empirische Coverage (Soll &ge; 80&nbsp;%), rechts die "
                "dafuer noetige Intervallbreite.",
            )
        )
        parts.append(f"""
<table><thead><tr><td>Methode</td><th>Horizont</th><th>Coverage</th><th>Median-Breite</th></tr></thead>
<tbody>{cov_rows}</tbody></table>
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Split-Conformal liegt nahe am Soll</strong> (68-76&nbsp;% gegenueber
80&nbsp;% nominal) - mit einer Zeile Code pro Seite und ohne Zusatzmodell.</li>
<li><strong>Quantil-Regression unterschaetzt Unsicherheit</strong> auf echten,
heterogenen Daten (58-69&nbsp;%) - die Baender sehen schlank aus, sind aber
zu eng. Faustregel: Quantile kalibrieren, Conformal garantiert
(unter Austauschbarkeit) approximativ.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e13
    if e13:
        ratio_lines = []
        for scen, d in e13["scenarios"].items():
            r = d["_ratio_vs_l2"]
            span = {
                m: f"{min(v.values()):.3f}-{max(v.values()):.3f}"
                for m, v in r.items()
                if m != "lgbm_l2"
            }
            ratio_lines.append(
                f"<tr><td>{scen}</td>"
                + "".join(f"<td>{span.get(m, '&ndash;')}</td>" for m in
                          ("lgbm_l1", "lgbm_huber", "lgbm_quantile50"))
                + "</tr>"
            )
        parts.append(f"""
<h2 id="e13">E13 &middot; Trainings-Objective: lohnt sich L1/Huber/Quantile?</h2>
<p>Vier Objectives, identischer Backtest (rolling origin), Metrik MAE -
dargestellt als Ratio zu L2 (=1). Zwei Dokumentationen: LightGBMs
Huber-Delta ist ABSOLUT (Default 0.9 kollabiert bei Labels &gt;&gt; 1 -
hier auf 2&times;Std skaliert); Quantile(0.5) IST mathematisch L1
(identische Baeume - die identischen Spalten bestaetigen das).</p>
<table><thead><tr><td>Daten</td><th>L1</th><th>Huber</th><th>Quantile(0.5)</th></tr></thead>
<tbody>{''.join(ratio_lines)}</tbody></table>
<div class="card finding">
<strong>Befund.</strong> Robuste Objectives bringen auf synthetischen,
gaussnahen DGPs nichts (bis ~26&nbsp;% <em>schlechter</em> beim Median-Ziel
auf Trend+Saison) und auf echten M4-Daten ~2-4&nbsp;% MAE-Vorteil fuer
L1/Quantile. Der L2-Default bleibt eine gute erste Wahl; Huber nur mit
skaliertem Delta einsetzen.
</div>""")

    # ------------------------------------------------------------------ e14
    if e14:
        mo = e14["mase_overall"]
        mase_rows = "".join(
            f"<tr><td>{k}</td><td{' class=hl' if k == min(mo, key=mo.get) else ''}>{fmt(float(v), 3)}</td></tr>"
            for k, v in sorted(mo.items(), key=lambda kv: kv[1])
        )
        parts.append("""
<h2 id="e14">E14 &middot; Foundation Models auf M4: Chronos-Bolt &amp; Chronos-2</h2>
<p>Dieselben Daten, dasselbe Fixed-Origin-Protokoll wie E11 - aber die
Foundation-Modelle laufen strikt Null-Shot (nur Inferenz, kein Training,
kein Fine-Tuning): Chronos-Bolt-Base (univariate, T5-Encoder-Decoder,
quantile heads) und Chronos-2 (universal In-Context-Learning).</p>
""")
        parts.append(
            fig(
                "e14_chronos_m4",
                "Foundation Models vs. trainierte Arme",
                "MAE je Horizont. Chronos-2 startet am kurzen Horizont vorn; "
                "Bolt ohne Kovariaten/Transformation verliert die Trendspur "
                "(inverses Horizontprofil).",
            )
        )
        parts.append(f"""
<h3>MASE je Arm</h3>
<table><thead><tr><td>Modell</td><th>MASE</th></tr></thead>
<tbody>{mase_rows}</tbody></table>
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Chronos-2 spielt sofort oben mit:</strong> MASE&nbsp;0.938 -
auf dem Niveau von Theta/AutoETS (0.932/0.936), vor allen LGBM-Armen;
am kurzen Horizont die beste Einzelprognose (MAE&nbsp;178 vs 184/186).
Null-Shot, ohne eine Zeile Training.</li>
<li><strong>Chronos-Bolt enttaeuscht auf diesem Panel</strong> (MASE&nbsp;1.66,
h=1: MAE&nbsp;730): univariate Level-Prognosen ohne Extrapolationshilfe
verlieren auf stark trendenden Serien die Spur - das inverses
Horizontprofil (schlechter bei h=1 als bei h=18) ist das Symptom.
Bolt ist fuer Geschwindigkeit optimiert, nicht fuer dieses Regime.</li>
<li><strong>Einordnung zu E11:</strong> Der Abstand klassisch-lokal vs.
global-ML schliesst sich durch Foundation-Modelle von oben - nicht durch
das trainierte LGBM von unten.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e15
    if e15:
        mm = e15["metrics"]
        rows = []
        for m in ("lgbm_levels", "lgbm_levels_x", "chronos-bolt-base", "chronos-2", "chronos-2-x"):
            if m not in mm:
                continue
            cells = "".join(
                f"<td>{fmt(float(mm[m][h]['mae']), 1)}</td>" for h in ("1", "6", "12", "18")
            )
            da = mm[m]["18"].get("dir_acc")
            da_s = fmt(100 * float(da), 0) + "&nbsp;%" if da == da and da is not None else "n.v."
            rows.append(f"<tr><td>{m}</td>{cells}<td>{da_s}</td></tr>")
        parts.append(f"""
<h2 id="e15">E15 &middot; Budgetpl&auml;ne als Zukunftskovariate: nutzt sie jemand?</h2>
<p>Synthesewelt wie E8 (Trend + Saison + persistenter Treiber <code>x</code>,
60 Serien); der Treiberpfad ist zum Prognosezeitpunkt <em>bekannt</em> -
Budgetplan-Semantik. LGBM bekommt ihn als Szenario-Feature (E4/E8-Muster),
Chronos-2 nativ als Zukunftskovariate, Chronos-Bolt kann ihn nicht sehen.
MAE je Horizont und Directional Accuracy bei h=18:</p>
<table><thead><tr><td>Modell</td><th>h=1</th><th>h=6</th><th>h=12</th><th>h=18</th><th>DirAcc h=18</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Trainiertes LGBM mit Plan bleibt vorn</strong> (h=18: MAE&nbsp;46.5,
~10-30&nbsp;% besser als seine eigene Ablation ohne Treiber - das E8-Muster
repliziert).</li>
<li><strong>Chronos-2 f&auml;ngt den Treiber Null-Shot gro&szlig;teils ein:</strong>
mit Kovariate schlaegt es seine eigene univariate Variante um ~8-15&nbsp;%
(h=18: 52.0 vs 58.4, DirAcc 0.683&rarr;0.767) und liegt am kurzen Horizont
sogar vor dem trainierten LGBM (h=1: 11.4 vs 12.4). Der Restabstand zum
LGBM+Szenario oeffnet sich erst ab h=12.</li>
<li><strong>Chronos-Bolt kann per Design nicht davon profitieren</strong> und
bliest das Niveau: inverses Profil wie in E14 (h=1: 103 vs naive 13.6),
nur im langen Horizont konkurrenzfaehig.</li>
<li><strong>Praxis-Fazit:</strong> Zukunftswissen ist der Hebel, der unsere
Studien durchgehend zeigen - Foundation-Modelle machen ihn jetzt
inferenz-only nutzbar, aber trainierte Modelle mit Szenario-Features
setzen ihn bei weitem Horizont immer noch praeziser um.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ literatur
    parts.append("""
<h2 id="literatur">Einordnung: was Praxis &amp; Literatur dazu sagen</h2>
<p>Die Befunde dieses Reports kreuzen wir mit drei Quellen-Kreisen - mit
Ergebnis: keine Widerspr&uuml;che, mehrere direkte Best&auml;tigungen,
zwei Inspirationen (E9, E10):</p>
<ul>
<li><strong>Globale Modelle gewinnen Panel-Wettbewerbe.</strong> Die
M5-Studie (Makridakis et&nbsp;al., IJF 2022): Alle Top-L&ouml;sungen sind
globale Modelle, LightGBM dominiert; Cross-Learning &uuml;ber Serien ist
der Kernvorteil. E1 st&uuml;tzt das (Panel-Learning als Rauschfilter); auf
echten M4-Serien ist der Vorsprung gegen&uuml;ber der schlichten Naive mit
der Default-Konfiguration aber klein und horizontabh&auml;ngig (E5) - die
M5-Dominanz kommt vermutlich aus genau den Hebeln, die E2/E6/E8 zeigen
(Zieltransformation, Features, Ensembles).
<a href="https://www.sciencedirect.com/science/article/pii/S0169207021001722">M5 accuracy competition (IJF)</a></li>
<li><strong>SHAP erkl&auml;rt das Modell, nicht die Welt.</strong>
Praxis-Darstellungen zur SHAP-Nutzung im Forecasting betonen: Zeitstruktur
muss in die Interpretation (Lag-Bl&ouml;cke aggregieren), und der
st&auml;rkste Anwendungsfall ist die Erkl&auml;rung von
Forecast-<em>Revisionen</em> zwischen zwei Origins. E10 setzt beides um -
und liefert mit dem bekannten DGP das fehlende St&uuml;ck: den
objektiven Vergleich gegen das wahre Signal.
<a href="https://www.analytical-software.de/en/time-series-forecasting-with-shap/">Time Series Forecasting with SHAP (HMS)</a></li>
<li><strong>Korrelierte Features spalten Attribution.</strong> Bekannte
Schw&auml;che von Baum-Importances: stark korrelierte Lags/Rollings teilen
sich den Kredit, Einzelwerte werden unzuverl&auml;ssig. E10 zeigt das
quantitativ: Die Steigung von SHAP(x) zerf&auml;llt schneller als das
wahre Signal - Attribution geh&ouml;rt immer trianguliert gelesen
(SHAP + Gain + Ablation + Intervention), nie einzeln.</li>
<li><strong>Zieltransformation vor Modellglanz.</strong> Nixtlas
mlforecast-Dokumentation macht Log-/Differenz-Transforms mit inverser
R&uuml;cktransformation zum Standardweg - genau die E2/E6-Empfehlung.
<a href="https://nixtlaverse.nixtla.io/mlforecast/">mlforecast docs (Nixtla)</a></li>
</ul>
""")

    # ------------------------------------------------------------------ synthese
    parts.append("""
<h2 id="synthese">Gegen&uuml;berstellung: welche Eigenschaft erzwingt welche Entscheidung?</h2>
<p>Die Studien im Verbund - jede Zeile eine Dateneigenschaft, daneben der
empirische Beleg und die Konsequenz f&uuml;r das Setup:</p>
<table>
<thead><tr><td>Eigenschaft</td><th>Beleg</th><th>Konsequenz</th></tr></thead>
<tbody>
<tr><td>Starker exponentieller Trend</td><td>E2, E6-stark</td><td>Log-Transformation + h-Schritt-Differenz als Label; sonst bis 23&times; Fehler bei h=18</td></tr>
<tr><td>Kein bis schwacher Trend</td><td>E6-kein/-leicht</td><td>Transformation unn&ouml;tig, Levels reichen - Komplexit&auml;t sparen</td></tr>
<tr><td>Additiver (linearer) Trend</td><td>E6-linear</td><td>YoY-Lags interpolieren ihn; Levels vertretbar, Log-Diffs nie schlechter</td></tr>
<tr><td>Saisonalit&auml;t stark</td><td>E1, E7-Galerie</td><td>Lag-12 + rollende Fenster tragen sie; Seasonal-Naive nur als Referenz</td></tr>
<tr><td>Saisonalit&auml;t fehlt/schwach</td><td>E7b-Raster</td><td>Empfehlungen bleiben gleich; Seasonal-Naive verliert ihren Anker komplett</td></tr>
<tr><td>Viele verwandte Serien (Panel)</td><td>E1, E5, E11 (M4)</td><td>Panel-Learning filtert Rauschen (+~18&nbsp;% gg. Einzel-Fits, E11); am 1-Schritt-Fenster ist die Naive hart - und klassische lokale Modelle bleiben auf klassischem M4 vorn</td></tr>
<tr><td>Unsicherheitsbaender geplant</td><td>E12</td><td>Split-Conformal um die Punktprognose haelt Coverage naeher am Soll als ungepruefte Quantil-Regression</td></tr>
<tr><td>Ausreisser-/Schwanzdruck</td><td>E13</td><td>L1/Quantile(0.5) ~2-4&nbsp;% MAE auf echten Daten; Huber nur mit Delta auf Labelskala (Default kollabiert)</td></tr>
<tr><td>Treiber-Intervention / Szenario</td><td>E4</td><td>Nur ein Modell mit kausalem Treiber + bekanntem Pfad trackt den Eingriff</td></tr>
<tr><td>Strukturbruch im Trend</td><td>E6-Umkehr</td><td>Direkt-Formulierung friert das alte Regime ein; Rollout/kurzes Retraining adaptiert</td></tr>
<tr><td>Hyperparameter-Druck</td><td>E9</td><td>Defaults halten (&le;6&nbsp;% Gewinn, bis &minus;16&nbsp;% Verlust bei Br&uuml;chen); Hebel sind Formulierung &amp; Features</td></tr>
<tr><td>Ausreisser-/Schwanzdruck</td><td>E13</td><td>L1/Quantile(0.5) ~2-4&nbsp;% MAE auf echten Daten; Huber nur mit Delta auf Labelskala (Default kollabiert)</td></tr>
<tr><td>Zukunftsbekannte Treiber + Null-Shot-Anforderung</td><td>E15</td><td>Chronos-2 nutzt Budgetplaene inferenz-only (~8-15&nbsp;%); trainiertes LGBM+Szenario bleibt bei weiten Horizonten vorn</td></tr>
</tbody></table>
<div class="card finding">
<strong>Der rote Faden.</strong> Kein Ergebnis widerspricht einem anderen -
die Effekte stapeln sich: <em>Formulierung zuerst</em> (Log-Diffs unter Trend),
<em>dann Features</em> (kausale Treiber, Kalender), <em>dann Strategie</em>
(direkt statt rekursiv; nach Br&uuml;chen Adaptierung). E8 zeigt in einem
DGP, der alle Eigenschaften zugleich hat: 35&nbsp;% weniger Fehler als rohe
Levels, wenn man alles kombiniert.
</div>
""")
    # ------------------------------------------------------------------ takeaways
    parts.append("""
<h2 id="takeaways">Empfehlungen f&uuml;rs Praxis-Playbook</h2>
<div class="card">
<ol>
<li><strong>Global, nicht pro Serie:</strong> Ein Modell &uuml;ber alle Serien
nutzt Querschnittsstruktur und ist ab ~50 Serien praktisch immer effizienter
(E1: Panel-Learning als Rauschfilter).</li>
<li><strong>Direct Multi-Horizont statt rekursiv:</strong> ein Booster pro
Horizont verhindert den Fehler-Schneeball (E6: 16&times; bei h=18). Kein
rekursives Vorgehen, wenn direkte Labels billig sind - und das sind sie.</li>
<li><strong>Ziel transformieren, nicht das Modell verbiegen:</strong> Log- und
Saisondifferenzen machen Trends extrapolierbar (E2: Faktor ~120 bei h=12).
Der gr&ouml;&szlig;te einzelne Genauigkeitshebel des gesamten Reports.</li>
<li><strong>Feature-Familien kombiniert einsetzen:</strong> Lags allein sind
schwach; Rolling-Stats + Kalender bringen die n&auml;chsten Spr&uuml;nge (E3).</li>
<li><strong>Treiber-Szenarien einplanen:</strong> Wenn f&uuml;hrende Gr&ouml;&szlig;en
(Budgets, Preise, Pl&auml;ne) f&uuml;r die Zukunft bekannt sind, geh&ouml;ren sie als
Szenario-Features ins Modell (E4: 49&nbsp;% &rarr; 93&nbsp;% Directional Accuracy
unter Intervention). Persistente oder geplante Treiber lohnen sich;
schnell vergessene AR(1)-Treiber nur am kurzen Horizont.</li>
<li><strong>Backtest-Hygiene:</strong> Trainingszeilen nur mit
<code>target_date &le; Fold-Ende</code>; Baselines rolling-origin aus demselben
Informationsstand wie das Modell (frozen Baselines haben E5 einst ~41&nbsp;%
Phantomvorsprung suggeriert); Metriken auf gemeinsamer Unterlage;
Directional Accuracy nur wo Richtung definiert ist.</li>
<li><strong>Erkl&auml;rung &ne; Prognoseg&uuml;te:</strong> Importance unter
Regime-Daten sagt nichts &uuml;ber kausale Richtigkeit. Interventionstests
(auch simulierte) sind der H&auml;rtetest (E4). Importance immer triangulieren:
SHAP + Gain + Ablation + Intervention (E10: Treiber bekommt ~3&nbsp;% Budget,
tr&auml;gt aber 24&nbsp;% des wahren Signals).</li>
<li><strong>Revisionen erkl&auml;ren, nicht verstecken:</strong> Wenn ein
Forecast sich zwischen zwei Origins &auml;ndert, ist die SHAP-Differenz der
zwei Erkl&auml;rungen die vollst&auml;ndige, additive Begr&uuml;ndung
(E10 D) - das Kommunikationsformat f&uuml;r Stakeholder.</li>
<li><strong>Hilfsspalten-Disziplin:</strong> Jede numerische Zusatzspalte in
der Supervised-Tabelle wird zum Feature, wenn die Exogen-Auswahl nicht
explizit gesetzt ist (E6-Debugging-Fund: <code>y_change</code> als Feature
invalidiert still die Vergleichsstudie).</li>
</ol>
</div>""")

    # ------------------------------------------------------------------ appendix
    parts.append("""
<h2 id="appendix">Anhang &amp; Reproduktion</h2>
<h3>Studien&uuml;bersicht</h3>
<table>
<thead><tr><td>Studie</td><th>Frage</th><th>Daten</th><th>Modelle</th></tr></thead>
<tbody>
<tr><td>E1</td><td>Wann gewinnt LGBM?</td><td>5 synthetische Szenarien</td><td>LGBM, SNaive, Naive</td></tr>
<tr><td>E2</td><td>Zieltransformation</td><td>1 Szenario, exponentiell</td><td>LGBM (4 Preps)</td></tr>
<tr><td>E3</td><td>Feature-Familien</td><td>DGP mit Treiber x</td><td>LGBM (6 Sets)</td></tr>
<tr><td>E4</td><td>Kausalit&auml;t/Intervention</td><td>DGP mit OU-Treiber</td><td>LGBM (3 Varianten)</td></tr>
<tr><td>E5</td><td>Realer Benchmark</td><td>M4, 400 Serien</td><td>LGBM, SNaive, Naive</td></tr>
<tr><td>E6</td><td>Level vs. Log-Diff, Rekursion, Trend-Regime inkl. Umkehr</td><td>5 Regime, exponentiell/linear/Bruch</td><td>LGBM (3 Varianten), SNaive</td></tr>
<tr><td>E7</td><td>Galerie: Verl&auml;ufe mit Prognosen</td><td>M4-Auswahl + Raster Regime &times; Saison</td><td>LGBM (3 Varianten), SNaive</td></tr>
<tr><td>E8</td><td>Alles kombiniert: Trend + Saison + Treiber, 2&times;2 Ans&auml;tze</td><td>1 realistisches DGP (alle Eigenschaften)</td><td>LGBM (4 Zellen), SNaive</td></tr>
<tr><td>E9</td><td>Lohnt Hyperparameter-Tuning?</td><td>5 Szenarien (E6/E8-DGPs, M4)</td><td>LGBM (Optuna, 40 Trials)</td></tr>
<tr><td>E10</td><td>Was treibt die Prognose? Attribution vs. wahres DGP</td><td>60 Serien &times; 168 Monate, bekanntes DGP</td><td>LGBM + TreeSHAP (nativ)</td></tr>
<tr><td>E11</td><td>Beste Formulierung vs. klassisch-lokal</td><td>M4, 400 Serien, fixed origin</td><td>LGBM (Levels/LogDiff/Ensemble/je-Serie), AutoETS, Theta, Baselines</td></tr>
<tr><td>E12</td><td>Prognoseintervalle: Quantil vs. Conformal</td><td>M4, 200 Serien, Log-Raum</td><td>Quantil-LGBM, Split-Conformal</td></tr>
<tr><td>E13</td><td>Objective-Ablation (L2/L1/Huber/Q50)</td><td>3 synthetische Szenarien + M4 (150)</td><td>LGBM mit Objective-Overrides</td></tr>
<tr><td>E14</td><td>Foundation Models vs. trainierte Arme (Null-Shot)</td><td>M4, 400 Serien, fixed origin</td><td>Chronos-Bolt, Chronos-2 + E11-Arme</td></tr>
<tr><td>E15</td><td>Budgetplan als Zukunftskovariate</td><td>Synthetik mit Treiber, 60 Serien</td><td>LGBM&plusmn;Szenario, Bolt, Chronos-2&plusmn;Kovariate</td></tr>
</tbody></table>
<h3>Ausf&uuml;hren</h3>
<pre><code>uv sync
uv run python studies/e1_scenarios.py
uv run python studies/e2_data_prep.py
uv run python studies/e3_feature_ablation.py
uv run python studies/e4_causal.py
uv run python studies/e5_m4.py
uv run python studies/e6_levels_vs_logdiff.py
uv run python studies/e7_gallery.py
uv run python studies/e8_combined.py
uv run python studies/e9_tuning.py
uv run python studies/e10_shap_drivers.py
uv run python studies/e11_m4_best.py
uv run python studies/e13_objective_ablation.py
uv run python studies/e14_chronos_m4.py
uv run python studies/e15_chronos_exog.py
uv run python studies/build_report.py   # diesen Report neu bauen</code></pre>
<p>Jede Studie schreibt <code>reports/results/&lt;name&gt;.json</code> und
Abbildungen nach <code>reports/assets/</code>. Der Report embeddet beides
selbstenthalten (Base64).</p>
<footer>Erzeugt aus den JSON-Ergebnissen in reports/results/ - alle Zahlen
und Abbildungen stammen aus den ausgef&uuml;hrten Experimenten dieses Repos.</footer>""")

    html = (
        "<!doctype html><html lang='de'><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<title>LightGBM Panel-Forecasting - Insight Report</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        "<header class='hero'><h1>LightGBM Panel-Forecasting</h1>"
        "<p class='sub'>Insight-Report: Szenarien, Datenaufbereitung, Features,"
        " Treiber-Attribution &amp; Tuning &middot; monatliche"
        " Serien &middot; Horizont 1-18 Monate</p></header>"
        + "".join(parts)
        + "</div></body></html>"
    )
    out = ROOT / "reports" / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report geschrieben: {out} ({out.stat().st_size / 1024:.0f} KB)")


def ratio_table(ratio: dict[str, dict[str, float]]) -> str:
    rows = ""
    for scenario, hs in ratio.items():
        cells = "".join(
            f"<td{' class=hl' if v < 1 else ' class=lo' if v > 1 else ''}>{v}</td>"
            for v in hs.values()
        )
        rows += f"<tr><td>{scenario}</td>{cells}</tr>"
    return (
        "<table><thead><tr><td>Szenario</td><th>h=1</th><th>h=6</th><th>h=12</th><th>h=18</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p style='color:var(--muted);font-size:13px'>Ratio &lt; 1: LGBM besser. "
        "Alle Werte aus reports/results/e1_scenarios.json.</p>"
    )


if __name__ == "__main__":
    main()
