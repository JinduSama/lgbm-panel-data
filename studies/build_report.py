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
        rows.append(
            f'<tr class="group"><td colspan="{len(models) + 1}">{value.upper()}</td></tr>'
        )
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
    parts: list[str] = []

    # ------------------------------------------------------------------ TOC
    parts.append(
        "<h2>Inhalt</h2>"
        '<nav class="toc"><ol>'
        "<li><a href='#tldr'>Kernbefunde</a></li>"
        "<li><a href='#method'>Methodik im Detail</a></li>"
        "<li><a href='#e1'>E1 &middot; Szenario-Raster</a></li>"
        "<li><a href='#e2'>E2 &middot; Datenaufbereitung</a></li>"
        "<li><a href='#e3'>E3 &middot; Feature-Ablation</a></li>"
        "<li><a href='#e4'>E4 &middot; Kausale Plausibilit&auml;t</a></li>"
        "<li><a href='#e5'>E5 &middot; M4-Benchmark</a></li>"
        "<li><a href='#e6'>E6 &middot; Level vs. Log-Diffs &amp; Rekursion</a></li>"
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
        mm = e6["metrics_on_levels"]
        if "recursive_logdiff" in mm and "direct_logdiff" in mm:
            r = mm["recursive_logdiff"].get("18", {}).get("mae")
            d = mm["direct_logdiff"].get("18", {}).get("mae")
            if r and d:
                e6_ratio = r / d
    e4_dir = None
    if e4:
        iv = {r["model"]: r["dir_acc"] for r in e4["intervention"]}
        e4_dir = iv.get("with_x_plan")

    kpis = []
    if e2_ratio:
        kpis.append((f"{e2_ratio:,.0f}×", "weniger Fehler durch Log-Saisondifferenzen statt Levels (h=12, E2)"))
    if e6_ratio:
        kpis.append((f"{e6_ratio:,.1f}×", "Rekursions-Strafe bei h=18: rekursiv vs. direkt auf Log-Diffs (E6)"))
    if e4_dir:
        kpis.append((f"{100 * e4_dir:.0f} %", "Directional Accuracy nach Intervention - nur mit Treiber-Pfad (E4)"))
    if e5:
        mase = e5["mase_overall"].get("lgbm")
        if mase:
            kpis.append((f"{mase:.2f}", "MASE auf 400 echten M4-Serien (&lt;1 schl&auml;gt saisonale Naive, E5)"))
    if kpis:
        parts.append(
            '<h2 id="tldr">Kernbefunde</h2><div class="kpis">'
            + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="t">{t}</div></div>' for v, t in kpis)
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
<p>Expanding Window: Fold k trainiert auf allen Zielen bis Stichtag
T<sub>k</sub> und testet auf die folgenden <code>step</code> Monate
(<code>step = max(Horizonte)</code>, nicht-&uuml;berlappende Testfenster):</p>
<pre><code>Historie:  |—————— Train ——————| Test |
Fold 1:    |—————— T1 —————————|  T1+step  |
Fold 2:    |—————————— T2 ————————|  T2+step  |
Fold 3:    |—————————————— T3 ———————|  T3+step  |  (T3+step = Datenende)</code></pre>
<p><strong>Metriken:</strong></p>
<ul>
<li><strong>MAE</strong> = Mittel |y&minus;&#375;| - robust, prim&auml;re Referenz</li>
<li><strong>RMSE</strong> = sqrt(Mittel (y&minus;&#375;)&sup2;) - bestraft Ausrei&szlig;er</li>
<li><strong>sMAPE</strong> = 200 % &times; Mittel |y&minus;&#375;| / (|y|+|&#375;|) - relativ, gr&ouml;&szlig;enunabh&auml;ngig</li>
<li><strong>Directional Accuracy</strong>: Anteil korrekter Richtungen
sign(&#375;&minus;y<sub>ref</sub>) = sign(y&minus;y<sub>ref</sub>) mit
y<sub>ref</sub> = letzter beobachteter Wert zum Forecast-Origin; Zeilen ohne
definierte Richtung (Bewegung 0) werden ausgeschlossen</li>
<li><strong>MASE</strong> (nur E5): MAE skaliert mit dem In-Sample-Fehler der
1-Schritt-saisonalen Naive (m=12) derselben Serie; &lt; 1 hei&szlig;t besser
als diese Referenz</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e1
    if e1:
        parts.append("""
<h2 id="e1">E1 &middot; Szenario-Raster: wann gewinnt LGBM?</h2>
<p>F&uuml;nf kontrollierte Datenerzeugungsprozesse (Trend &times; Saisonalit&auml;t &times;
Rauschen) gegen die Baselines Naive und Seasonal-Naive. Je Szenario ein
eigenes Panel; identischer Backtest (3 Folds &times; 18 Monate).</p>
""")
        parts.append(setup_box(**{
            "Serien": "50 je Szenario", "Länge": "132 Monate",
            "Horizonte": "1/6/12/18", "Folds": "3", "Boosting": "300 Runden",
        }))
        parts.append(fig(
            "e1_scenario_grid", "MAE je Szenario und Modell",
            "Jedes Panel zeigt MAE &uuml;ber den Horizont f&uuml;r die drei Modelle. "
            "Je gr&ouml;&szlig;er der Abstand der blauen (LGBM) zur orangen Linie "
            "(Seasonal-Naive), desto gr&ouml;&szlig;er der LGBM-Vorteil.",
        ))
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
gewinnt 8-15&nbsp;% (Ratio 0.85-0.92) - Panel-Learning als Rauschfilter.</li>
<li><strong>Trend ist der gr&ouml;&szlig;te Hebel:</strong> Bei exponentiellem Wachstum
bleibt Seasonal-Naive 12 Monate zur&uuml;ck; LGBM erreicht nur 26-32&nbsp;% deren
Fehler, weil Year-over-Year-Differenzen das Wachstum extrapolieren.</li>
<li>Auch ohne Saisonalit&auml;t gewinnt LGBM auf strukturllosen Serien (Ratio
0.74-0.80): die Serien-ID als kategoriales Feature lernt unterschiedliche
Niveaus.</li>
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
        parts.append(setup_box(**{
            "Serien": "60", "Länge": "132 Monate", "Horizonte": "1/6/12",
            "Folds": "3", "Wachstum": "1.5-3.5 %/Monat", "Saison": "15-35 abs.",
        }))
        parts.append(fig(
            "e2_data_prep", "MAE und Richtungsguete je Transformation",
            "Links MAE (je niedriger desto besser), rechts Directional Accuracy "
            "(0.5 = M&uuml;nzwurf, 1.0 = perfekte Richtung). Achte auf den "
            "Niveaunterschied zwischen <em>levels</em> und den drei "
            "transformierten Varianten.",
        ))
        parts.append("<h3>Vollst&auml;ndige Metriken je Transformation</h3>")
        parts.append(metrics_table(s, values=("mae", "rmse", "dir_acc"), digits=2,
                                   lower_is_better={"dir_acc": False}))
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
        parts.append(setup_box(**{
            "Serien": "60", "Länge": "132 Monate", "Horizonte": "1/6/12/18",
            "Folds": "3", "Treiber": "AR(1), φ=0.7, β∈[1.8,2.6]",
        }))
        parts.append(fig(
            "e3_feature_ablation", "Feature-Ablation MAE und Importance",
            "Links: MAE-Kurven der sechs Feature-Sets - je tiefer, desto besser; "
            "die Reihenfolge der Farben folgt der Ablationskette. Rechts: "
            "Gain-Anteile je Feature-Familie des vollst&auml;ndigen Modells bei h=12.",
        ))
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
        parts.append(setup_box(**{
            "Serien": "40", "Länge": "150 Monate", "Intervention": "Monat 132, x → 35 %",
            "Origin": "T=132", "Fenster": "18 Monate", "Boosting": "300 Runden",
        }))
        parts.append(fig(
            "e4_causal_intervention", "Interventionsexperiment",
            "Links (Beispielserie S00): schwarze Linie = Wahrheit nach "
            "Budget-Senkung, grau gestrichelt = Welt ohne Senkung. Die rote "
            "Prognose (lag_only) bleibt oben im alten Regime, t&uuml;rkis "
            "(with_x_plan) folgt dem Einbruch. Mitte: MAE je Horizont im "
            "Interventionsfenster. Rechts: Gain-Anteile beider Modelle.",
        ))
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
<p>{e5['n_series']} zuf&auml;llig gezogene M4-Monatsserien (Wettbewerb: Monats-
daten, Horizont 18), 2 Folds &times; 18 Monate Testfenster, identische Pipeline
wie die synthetischen Studien.</p>
""")
        parts.append(setup_box(**{
            "Serien": str(e5["n_series"]), "Folds": "2 × 18 Monate",
            "Horizonte": "1/6/12/18", "Boosting": "400 Runden",
        }))
        parts.append(fig(
            "e5_m4_benchmark", "M4 Benchmark",
            "Links MAE, Mitte sMAPE (beide je Horizont), rechts MASE als "
            "Gesamtkennzahl. Auf echten Daten sind die Niveaus gro&szlig; und "
            "serienheterogen - die <em>Relativordnung</em> der Modelle ist die "
            "Aussage, nicht der absolute Wert.",
        ))
        parts.append("<h3>Metriken je Horizont</h3>")
        parts.append(metrics_table(e5["metrics"], values=("mae", "smape"), digits=2))
        mo = e5["mase_overall"]
        mase_rows = "".join(
            f"<tr><td>{k}</td><td{' class=hl' if v == min(mo.values()) else ''}>{fmt(float(v), 3)}</td></tr>"
            for k, v in sorted(mo.items(), key=lambda kv: kv[1])
        )
        parts.append(
            "<h3>MASE (gegen In-Sample-Seasonal-Naive skaliert)</h3>"
            "<div class='card goodbox'><p>MASE &lt; 1 bedeutet: besser als die"
            " saisonale Naive auf dem eigenen Historieniveau. <strong>Caveat:</strong>"
            " MASE mittelt Verh&auml;ltnisse pro Serie - stark trendende Serien haben"
            " gro&szlig;e In-Sample-Nenner und dominieren das Bild; deshalb sind die"
            " MAE-/sMAPE-Charts oben die prim&auml;re Referenz.</p></div>"
            f"<table><thead><tr><td>Modell</td><th>MASE</th></tr></thead>"
            f"<tbody>{mase_rows}</tbody></table>"
        )
        imp = e5.get("importance_lgbm_h18_top10") or {}
        if imp:
            imp_rows = "".join(
                f"<tr><td>{k}</td><td>{fmt(100 * v, 1)}&nbsp;%</td></tr>"
                for k, v in imp.items()
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
<strong>Befunde.</strong>
<ul>
<li><strong>Kurzer Horizont: LGBM klar vorne</strong> (h=1: MAE 291 vs 496/507
- ~41&nbsp;% besser als beide Baselines).</li>
<li><strong>Langer Horizont auf Levels: Seasonal-Naive holt auf</strong>
(h=18: 496 vs 602). Exakt das E2/E6-Muster - ohne Zieltransformation
kann LGBM Trendwachstum &uuml;ber 18 Monate nicht extrapolieren. Die
Konsequenz aus E2 (Log-Saisondifferenzen) ist hier die offene Verbesserung.</li>
<li><strong>Directional Accuracy:</strong> LGBM trifft die Richtung deutlich
h&auml;ufiger als die Baselines (siehe results/e5_m4.json).</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e6
    if e6:
        mm = e6["metrics_on_levels"]
        parts.append("""
<h2 id="e6">E6 &middot; Level vs. Log-Differenzen: direkte Formulierung vs. Rekursion</h2>
<p>Die Kernfrage: <em>Sind direkte Level-Forecasts besser als
Log-Differenz-Forecasts - und was kostet die Rekursivit&auml;t der
Differenzen?</em> Vier Varianten auf identischem DGP (starker
Exponentialtrend + Saison, wie E2, aber Horizonte bis 18 und 1-Monats-
Aufl&ouml;sung der Kurven):</p>
<ul>
<li><strong>direct_level</strong>: ein LGBM pro Horizont auf rohen Levels</li>
<li><strong>seasonal_naive</strong>: Referenz-Baseline</li>
<li><strong>direct_logdiff</strong>: ein LGBM pro Horizont auf der
h-Schritt-&Auml;nderung <code>log y[t+h] &minus; log y[t]</code>;
Rekonstruktion <code>exp(log y[t] + pred)</code> - <strong>keine
Rekursion</strong>, der Anker ist beobachtet</li>
<li><strong>recursive_logdiff</strong>: <em>ein</em> Modell f&uuml;r
1-Schritt-&Auml;nderungen, das f&uuml;r h Schritte weiterspielt und Lags/Rolling
aus den eigenen Prognosen fortgeschreibt</li>
</ul>
""")
        parts.append(setup_box(**{
            "Serien": "60", "Länge": "144 Monate", "Horizonte": "1-18 (jede Stufe)",
            "Folds": "3 × 18 Monate", "Boosting": "300 Runden",
            "Rekursiv-Features": "Lags 1/2/3/6/12, roll3/12-mean, month",
        }))
        parts.append(fig(
            "e6_levels_vs_logdiff", "Level vs Log-Diff",
            "Links MAE auf Levels, logarithmische Y-Achse (die Unterschiede sind "
            "zu gro&szlig; f&uuml;r eine lineare Skala). Rechts Directional Accuracy. "
            "Die rote Kurve (rekursiv) w&auml;chst fast perfekt linear - das ist "
            "die Fehler-Akkumulation sichtbar gemacht.",
        ))
        parts.append("<h3>Metriken auf Levels (alle Horizonte)</h3>")
        parts.append(metrics_table(mm, values=("mae", "dir_acc"), digits=2,
                                   lower_is_better={"dir_acc": False}))
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Direct auf Levels ist durchgehend schwach</strong> (MAE ~370
flach &uuml;ber alle Horizonte): das Extrapolationsproblem aus E2, jetzt
bei feiner Horizontaufl&ouml;sung. Die Fehler kommen nicht vom Horizont,
sondern vom Level-Format.</li>
<li><strong>Rekursiv auf Log-Diffs startet stark und verliert linear:</strong>
MAE 11 (h=1) &rarr; 233 (h=18), Zuwachs ~12.7 pro Monat. Jede Stufe
konditioniert auf den Unscha&#772;rfen der vorherigen Prognosen; kleine
systematische Abweichungen in den Diffs compounding &uuml;ber 18 Stufen.
Das ist die quantifizierte Rekursions-Strafe: <strong>~16&times; gegen&uuml;ber
direkt bei h=18</strong>.</li>
<li><strong>Direkt auf Log-Diffs gewinnt &uuml;berall</strong> (MAE 5.1 &rarr; 14.6,
Dir. Acc &ge; 97&nbsp;%): die h-Schritt-Aenderung als Label bekommt die
Rekursion umgangen - ein Modell pro Horizont springt direkt zum Ziel, und
der Rekonstruktions-Anker (letzter beobachteter Wert) ist immer real.</li>
<li><strong>Seasonal-Naive ist auf diesem DGP chancenlos</strong> (MAE ~995,
Dir. Acc teils unter 0.1): 12 Monate hinter einem exponentiellen Trend
zur&uuml;ck zu bleiben ist die teuerste m&ouml;gliche Strategie.</li>
</ul>
</div>
<div class="card warn">
<strong>Praxis-Konsequenz.</strong> Differenzen bilden ist die richtige Idee -
aber als <em>direkte</em> h-Schritt-Formulierung, nicht als rekursives
1-Schritt-Modell. Kosten: mehrere Booster statt einem (vernachl&auml;ssigbar).
Die Kombination aus E2 + E6 ist das Rezept f&uuml;r trendende monatliche Serien:
<code>log</code>-Transformation, h-Schritt-Log-Differenz als Label, ein
Booster pro Horizont, Rekonstruktion aus dem beobachteten Anker.
</div>""")

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
<code>target_date &le; Fold-Ende</code>; Directional Accuracy immer gegen den
letzten beobachteten Wert; MASE-Verh&auml;ltnisse mit Vorsicht genie&szlig;en (E5).</li>
<li><strong>Erkl&auml;rung &ne; Prognoseg&uuml;te:</strong> Importance unter
Regime-Daten sagt nichts &uuml;ber kausale Richtigkeit. Interventionstests
(auch simulierte) sind der H&auml;rtetest (E4).</li>
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
<tr><td>E6</td><td>Level vs. Log-Diff, Rekursion</td><td>1 Szenario, exponentiell</td><td>LGBM (3 Varianten), SNaive</td></tr>
</tbody></table>
<h3>Ausf&uuml;hren</h3>
<pre><code>uv sync
uv run python studies/e1_scenarios.py
uv run python studies/e2_data_prep.py
uv run python studies/e3_feature_ablation.py
uv run python studies/e4_causal.py
uv run python studies/e5_m4.py
uv run python studies/e6_levels_vs_logdiff.py
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
        " kausale Plausibilit&auml;t &amp; Rekursions-Kosten &middot; monatliche"
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
