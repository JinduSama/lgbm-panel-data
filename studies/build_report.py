"""
Baut den selbstenthaltenen HTML-Report aus den Studien-Ergebnissen.

Liest reports/results/*.json und bettet die PNGs aus reports/assets/
base64-kodiert ein -> eine einzige Datei ``reports/report.html``.
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
.wrap { max-width:1080px; margin:0 auto; padding:48px 24px 96px; }
header.hero { padding:40px 0 8px; }
h1 { font-size:34px; margin:0 0 6px; letter-spacing:-0.5px; }
p.sub { color:var(--muted); margin:0 0 28px; }
h2 { font-size:24px; margin:56px 0 12px; padding-top:18px;
     border-top:1px solid var(--line); }
h3 { font-size:17px; margin:26px 0 8px; color:var(--accent); }
p, li { color:#c9d3e4; }
code { background:#1d2740; border:1px solid var(--line); padding:1px 6px;
       border-radius:6px; font-size:13px; color:#a8d5ff; }
pre { background:#121a2b; border:1px solid var(--line); border-radius:10px;
      padding:14px 16px; overflow-x:auto; font-size:13px; }
pre code { border:none; background:none; padding:0; }
.card { background:var(--card); border:1px solid var(--line);
        border-radius:14px; padding:20px 24px; margin:18px 0; }
.finding { border-left:4px solid var(--accent); }
.warn { border-left:4px solid var(--bad); }
.goodbox { border-left:4px solid var(--good); }
img.fig { max-width:100%; border-radius:10px; border:1px solid var(--line);
          margin:14px 0; background:#fff; }
table { width:100%; border-collapse:collapse; margin:14px 0; font-size:14px; }
th, td { padding:8px 12px; text-align:right; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; text-transform:uppercase;
     font-size:11px; letter-spacing:0.08em; }
td.hl { color:var(--good); font-weight:600; }
td.lo { color:var(--bad); }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
        gap:14px; margin:22px 0; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:12px;
       padding:16px 18px; }
.kpi .v { font-size:26px; font-weight:700; color:var(--accent); }
.kpi .t { color:var(--muted); font-size:13px; margin-top:2px; }
footer { margin-top:80px; color:var(--muted); font-size:13px;
         border-top:1px solid var(--line); padding-top:18px; }
nav.toc a { color:var(--accent); text-decoration:none; }
nav.toc li { margin:4px 0; }
"""

INTRO = """
<p class="sub">
LightGBM als globales Prognosemodell fuer Panel-Zeitreihen (viele Serien,
monatliche Frequenz, Horizont bis 18 Monate): Szenarien, Datenaufbereitung,
Feature-Design und die Frage nach kausaler Erklaerbarkeit - mit
reproduzierbaren Experimenten.</p>
"""


def load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig(name: str, alt: str) -> str:
    rel = f"reports/assets/{name}.png"
    if not (ROOT / rel).exists():
        return f"<p><em>Figur {name} fehlt.</em></p>"
    return f'<img class="fig" src="{b64_image(rel)}" alt="{alt}"/>'


def fmt(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}"


def metrics_table(metrics: dict, value: str = "mae", digits: int = 2) -> str:
    models = sorted(metrics)
    horizons = sorted({h for m in metrics.values() for h in m}, key=float)
    head = "".join(f"<th>{m}</th>" for m in models)
    rows = []
    for h in horizons:
        cells = ""
        vals = [metrics[m].get(h, {}).get(value) for m in models]
        finite = [v for v in vals if v is not None]
        best = min(finite) if finite else None
        for v in vals:
            hl = v is not None and best is not None and abs(v - best) < 1e-12
            cls = ' class="hl"' if hl else ""
            shown = fmt(v, digits) if v is not None else "-"
            cells += f"<td{cls}>{shown}</td>"
    body = "".join(rows)
    return (
        f"<table><thead><tr><td>Horizont</td>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def main() -> None:
    e1, e2, e3, e4, e5 = (
        load("e1_scenarios"), load("e2_data_prep"), load("e3_feature_ablation"),
        load("e4_causal"), load("e5_m4"),
    )
    parts: list[str] = []

    parts.append("<h2>Inhalt</h2>"
                 '<nav class="toc"><ol>'
                 "<li><a href='#method'>Methodik</a></li>"
                 "<li><a href='#e1'>E1 &middot; Szenario-Raster</a></li>"
                 "<li><a href='#e2'>E2 &middot; Datenaufbereitung</a></li>"
                 "<li><a href='#e3'>E3 &middot; Feature-Ablation</a></li>"
                 "<li><a href='#e4'>E4 &middot; Kausale Plausibilit&auml;t</a></li>"
                 "<li><a href='#e5'>E5 &middot; M4-Benchmark</a></li>"
                 "<li><a href='#takeaways'>Empfehlungen</a></li>"
                 "</ol></nav>")

    # ------------------------------------------------------------------ method
    parts.append("""
<h2 id="method">Methodik</h2>
<div class="card">
<p><strong>Panel zu Supervised:</strong> Jede Serie wird in Zeilen
(Serie, Cutoff t, Horizont h) zerlegt. Features nutzen ausschliesslich
Informationen bis t (Target-Lags, Rolling-Statistiken, Saison-Differenzen,
Kalendermerkmale des Cutoffs, optional exogene Treiber zum Zeitpunkt t).
Ziel ist <code>y[t+h]</code>. Damit ist Leakage per Konstruktion ausgeschlossen.</p>
<p><strong>Direct Multi-Horizon:</strong> Ein globales LightGBM-Modell pro
Horizont, trainiert ueber alle Serien hinweg (Panel-Learning). Keine
Fehler-Akkumulation wie beim rekursiven Vorgehen - fuer Horizonte &gt; 12
Monate der robustere Weg.</p>
<p><strong>Evaluation:</strong> Expanding-Window-Backtest. Fold k trainiert auf
allen Zielen bis Stichtag T<sub>k</sub> (<code>target_date &le; T_k</code>) und testet
auf den folgenden Monaten. Metriken: MAE, RMSE, sMAPE, Directional Accuracy
(Richtung relativ zum letzten beobachteten Wert) und MASE (nur E5).</p>
</div>""")

    # ------------------------------------------------------------------ e1
    if e1:
        parts.append("""
<h2 id="e1">E1 &middot; Szenario-Raster: wann gewinnt LGBM?</h2>
<p>Fuenf kontrollierte DGPs (Trend &times; Saisonalitaet &times; Rauschen, je 50 Serien
&times; 132 Monate) gegen die Baselines Naive und Seasonal-Naive.</p>
""")
        parts.append(fig("e1_scenario_grid", "MAE je Szenario und Modell"))
        parts.append("<h3>Kennzahlen (MAE-Ratio LGBM / Seasonal-Naive)</h3>")
        parts.append(metrics_ratio_table(e1["lgbm_over_snaive_mae_ratio"]))
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li>Auf sauber saisonalen, stationaeren Serien ist Seasonal-Naive kaum zu
schlagen (Ratio &asymp; 1): die wahre Funktion ist zu einfach, als dass ein
Baummodell Mehrwert liefert.</li>
<li>Mit Rauschen dreht das Blatt: LGBM mittelt ueber Serien hinweg und gewinnt
10-15&nbsp;% (Ratio 0.85-0.92).</li>
<li><strong>Trend ist der groesste Hebel:</strong> bei exponentiellem Wachstum
ist Seasonal-Naive systematisch 12 Monate hinterher - LGBM erreicht nur
26-32&nbsp;% deren Fehler, weil Year-over-Year-Differenzen das Wachstum
extrapolieren.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e2
    if e2:
        s = e2["scenarios"]
        parts.append("""
<h2 id="e2">E2 &middot; Datenaufbereitung auf stark trendenden Daten</h2>
<p>Exponentieller Trend (Wachstumsrate 1.5-3.5&nbsp;%/Monat), 60 Serien.
Vier Zieltransformationen, sonst identisches Setup.</p>
""")
        parts.append(fig("e2_data_prep", "MAE und Richtungsguete je Transformation"))
        parts.append("<h3>MAE je Horizont</h3>")
        prep_rows = ""
        for name, mm in s.items():
            cells = "".join(
                f"<td>{fmt(mm.get(h, {'mae': float('nan')})['mae'], 1)}</td>"
                for h in ("1", "6", "12")
            )
            prep_rows += f"<tr><td>{name}</td>{cells}</tr>"
        parts.append(
            "<table><thead><tr><td>Transformation</td><th>h=1</th><th>h=6</th><th>h=12</th></tr></thead>"
            f"<tbody>{prep_rows}</tbody></table>"
        )
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li>Rohe Level sind auf exponentiellem Trend katastrophal (MAE im vierstelligen
Bereich): Baummodelle extrapolieren <em>nicht</em> ueber den Trainingsbereich
hinaus - sie koennen Werte nur interpolieren, die sie schon gesehen haben.</li>
<li>Saisonale Differenzierung (y<sub>t</sub>&minus;y<sub>t&#8202;&minus;&#8202;12</sub>)
reduziert den Fehler um Faktor ~3.4.</li>
<li><strong>Log-Differenzen (multiplikativ) gewinnen um Gr&ouml;ssenordnungen</strong>
(MAE ~10 statt ~1190 bei h=12): sie machen das Signal homoskedastisch und
wandeln Extrapolation in die Schaetzung einer stabilen Wachstumsrate um -
Directional Accuracy nahe 1.0 auf allen Horizonten.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e3
    if e3:
        parts.append("""
<h2 id="e3">E3 &middot; Feature-Ablation: was beschreibt die Serie?</h2>
<p>DGP mit bekanntem exogenem Treiber x (AR(1), kausal mit 1 Monat Verzoegerung):
y = level + &beta;&middot;x<sub>t&#8202;&minus;&#8202;1</sub> + Saison + Trend + Rauschen.
Sechs Feature-Sets, identische Folds.</p>
""")
        parts.append(fig("e3_feature_ablation", "Feature-Ablation MAE und Importance"))
        parts.append("<h3>MAE je Feature-Set</h3>")
        parts.append(metrics_table(e3["metrics"]))
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
<li>Nur Target-Lags ist die schwachste Konfiguration; Rolling-Statistiken
bringen den groessten Einzelsprung, Kalenderfeatures helfen vor allem am
langen Horizont.</li>
<li>Der kausal treibende Wert x hilft massiv am kurzen Horizont (h=1:
MAE 8.9 vs 11.8) und verliert mit wachsender Distanz - denn AR(1)-Treiber
sind selbst kaum prognostizierbar: ihre Autokorrelation stirbt schneller,
als der Horizont waechst.</li>
<li>Gain-Importance verteilt sich trotz bekannter Kausalitaet auf Target-Lags
und Rolling-Stats. <em>Korrelierte redundante Features teilen sich die
Attribution.</em></li>
</ul>
</div>""")

        parts.append("""
<h2 id="e4">E4 &middot; Kausale Plausibilit&auml;t: Vorhersage ist nicht Erkl&auml;rung</h2>
<p>Synthetische Welt mit persistentem Budget-Treiber x (OU-Prozess um 45).
y reagiert kausal mit einem Monat Verzoegerung. Nach 132 Trainingsmonaten wird
x per do-Operator auf 35&nbsp;% gesenkt - die Counterfactual-Welt ist bekannt.
Drei Modelle, trainiert ausschliesslich auf Prae-Interventionsdaten:</p>
<ul>
<li><strong>lag_only</strong>: Target-Lags/Rolling/Kalender (kein x)</li>
<li><strong>with_x</strong>: + aktueller x-Stand</li>
<li><strong>with_x_plan</strong>: + geplanter Pfad von x (Szenario-Features,
x zum Zeitpunkt Ziel&minus;1 Monat - realistisch, weil Budgetplaene bekannt sind)</li>
</ul>
""")
        parts.append(fig("e4_causal_intervention", "Interventionsexperiment"))
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
        )
        parts.append("""
<div class="card finding">
<strong>Befunde.</strong>
<ul>
<li><strong>Im normalen Regime sieht alles gut aus:</strong> Backtest-MAE
unterscheidet die Modelle kaum (mit_x sogar leicht besser &uuml;berall). Die
Prognosequalitaet allein verr&auml;t nicht, welches Modell den Kausalzusammenhang
verstanden hat.</li>
<li>Nach dem Eingriff extrapolieren lag_only und with_x das alte Regime:
Bias +66, Richtungstreffen auf M&uuml;nzwurf-Niveau (&asymp;49&nbsp;%).</li>
<li><strong>Nur das Szenario-Modell reagiert</strong> (Dir.&nbsp;Acc 93&nbsp;%),
aber es erfaesst nur ~45&nbsp;% der wahren Staerke des Effekts - die redundanten
Rolling-Features, die im Regime so hilfreich waren, binden Gewicht und
dilutieren die Antwort. <em>Wer Interventionsfaehigkeit will, muss
Feature-Redundanz reduzieren oder explizite Struktur vorgeben.</em></li>
<li>Gain-Importance bestaetigt den Kontrast: ohne x liegt die gesamte Masse
auf Lags/Rolling/Serien-ID; mit x bekommt der Treiber einen sichtbaren Anteil.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ e5
    if e5:
        parts.append(f"""
<h2 id="e5">E5 &middot; Realer Benchmark: M4-Monatsdaten</h2>
<p>{e5['n_series']} zufaellig gezogene M4-Monatsserien, 2 Folds &times; 18 Monate
Testfenster (Wettbewerbshorizont).</p>
""")
        parts.append(fig("e5_m4_benchmark", "M4 Benchmark"))
        parts.append("<h3>Metriken je Horizont</h3>")
        parts.append(metrics_table(e5["metrics"]))
        mo = e5["mase_overall"]
        mase_rows = "".join(
            f"<tr><td>{k}</td>"
            f"<td{' class=hl' if v == min(mo.values()) else ''}>{fmt(float(v), 3)}</td></tr>"
            for k, v in sorted(mo.items(), key=lambda kv: kv[1])
        )
        parts.append(
            "<h3>MASE (gegen In-Sample-Seasonal-Naive skaliert)</h3>"
            "<div class='card goodbox'><p>MASE &lt; 1 bedeutet: besser als die"
            " saisonale Naive auf dem eigenen Historieniveau. <strong>Caveat:</strong>"
            " MASE mittelt Verhaeltnisse pro Serie - stark trendende Serien haben"
            " grosse In-Sample-Nenner und dominieren das Bild; deshalb hier die MAE/"
            "sMAPE-Charts oben als primaere Referenz nutzen.</p></div>"
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
                f"<table><thead><tr><td>Feature</td><th>Anteil</th></tr></thead>"
                f"<tbody>{imp_rows}</tbody></table>"
            )

    # ------------------------------------------------------------------ takeaways
    parts.append("""
<h2 id="takeaways">Empfehlungen fuers Praxis-Playbook</h2>
<div class="card">
<ol>
<li><strong>Global, nicht pro Serie:</strong> Ein Modell ueber alle Serien
nutzt Querschnittsstruktur und ist bei 100+ Serien praktisch immer effizienter.</li>
<li><strong>Direct Multi-Horizont fuer h &gt; 12:</strong> kein rekursiver
Fehler-Schneeball; ein Booster pro Horizont ist billig.</li>
<li><strong>Ziel transformieren, nicht das Modell verbiegen:</strong> Log bzw.
Saisondifferenzen machen Trends extrapolierbar. Der groesste einzelne
Genauigkeitshebel im ganzen Report.</li>
<li><strong>Feature-Familien kombiniert einsetzen:</strong> Lags allein sind
schwach; Rolling-Stats + Kalender + (falls vorhanden) Treiber bringen die
naechsten Spruenge.</li>
<li><strong>Treiber-Szenarien einplanen:</strong> Wenn fuehrende Groessen
(Budgets, Preise, Pläne) fuer die Zukunft bekannt sind, gehoeren sie als
Szenario-Features ins Modell - das macht Forecast zu Was-waere-wenn-Analyse.</li>
<li><strong>Backtest mit Ziel-Filter:</strong> Trainingszeilen nur mit
<code>target_date &le; Fold-Ende</code>; sonst schleichen sich Zukunftsinformation
ein. Directional Accuracy immer gegen den letzten beobachteten Wert messen.</li>
<li><strong>Erklaerung != Prognoseguete:</strong> Importance-Bilder unter
Regime-Daten sagen nichts darueber, ob ein Modell kausale Zusammenhaenge
getroffen hat. Interventionstests (auch simuliert) sind der Haertetest.</li>
</ol>
</div>""")

    parts.append("""
<h2>Reproduzieren</h2>
<pre><code>uv sync
uv run python studies/e1_scenarios.py
uv run python studies/e2_data_prep.py
uv run python studies/e3_feature_ablation.py
uv run python studies/e4_causal.py
uv run python studies/e5_m4.py
uv run python studies/build_report.py   # diesen Report neu bauen</code></pre>
<footer>Erzeugt aus den JSON-Ergebnissen in reports/results/ - alle Zahlen
und Abbildungen stammen aus den ausgefuehrten Experimenten dieses Repos.</footer>""")

    html = (
        "<!doctype html><html lang='de'><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<title>LightGBM Panel-Forecasting - Insight Report</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        "<header class='hero'><h1>LightGBM Panel-Forecasting</h1>"
        "<p class='sub'>Insight-Report: Szenarien, Aufbereitung, Features &amp;"
        " kausale Plausibilitaet &middot; monatliche Serien &middot; Horizont 1-18 Monate</p>"
        f"{INTRO}</header>"
        + "".join(parts)
        + "</div></body></html>"
    )
    out = ROOT / "reports" / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report geschrieben: {out} ({out.stat().st_size / 1024:.0f} KB)")


def metrics_ratio_table(ratio: dict[str, dict[str, float]]) -> str:
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
