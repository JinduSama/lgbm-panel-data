# Plan: LGBM Panel-Data Experiments

## Kontext
Wir haben in `lgbm_for_panel_data_forecasting.md` über LGBM für Panel/Time-Series
Forecasting diskutiert (monatliche Zeitreihen, Forecast Horizon >12 Monate).
Jetzt: Experiments mit Metriken und Graphen in Python aufbauen.

## Ziel
- Experiments-Pipeline für LGBM Panel-Forecasting
- Metriken: MAE, RMSE, sMAPE, directional accuracy
- Graphen: Vorhersage vs. Realität, Feature-Importance, Backtest-Metriken über Zeit
- Datensatz: Kaggle suchen, sonst synthetische Daten

## Architektur (src/lgbm_panel/)
- `data/`          -> Datensätze (synthetisch generiert + ggf. Kaggle)
- `features/`      -> Feature-Engineering (lags, rolling, time, cross-sectional)
- `metrics/`       -> Metriken-Berechnung
- `plotting/`      -> Visualisierungen
- `strategies/`    -> Recursive vs Direct Multi-Horizon
- `experiments/`   -> Experiment-Orchestrierung

## Schritte
1. Fallback-Plan als Markdown (dieses File)
2. Skill erstellen: Aufgaben stark in Chunks zerlegen + öfter kompaktieren (RAM-Schutz)
3. Datensatz: Kaggle (Mcompetitions/M4-methods -> 404) -> M4-Methoden via API gefunden.
   - `Mcompetitions/M4-methods/Dataset/Train/Monthly-train.csv` (91MB, 3000 Serie, wide format)
   - `Mcompetitions/M4-methods/Dataset/M4-info.csv` (M4id, category, Frequency, Horizon, SP, StartingDate)
   - Entscheidung: synthetische Daten (langsam, kontrollierbar) für schnelle Experiments + M4 als realer Test.
4. Feature-Engineering-Modul
5. LGBM-Strategien (recursive/direct)
6. Metriken
7. Graphen
8. Experiments ausführen & validieren

## Wichtige Fakten (aus Guide)
- LGBM = Regression mit Lags. Forecast als supervised learning umformulieren.
- Zwei Strategien: Recursive (ein Modell, Fehler summieren) vs Direct (ein Modell pro Horizon).
- Für Horizon >12 Monate: Direct bevorzugt (kein Fehler-Akkumulieren).
- Leakage vermeiden: nur vergangene Werte als Features.
- Panel: eigene Lags + cross-sectional Aggregation (leakage-achtend).
- Tools: sktime (make_reduction), darts (LGBMModel), mlforecast, raw lightgbm.
- Metriken: MAE, RMSE, sMAPE, directional accuracy.
- Evaluation: expanding-window Backtest.

## Nächster Fallback-Point
- [x] Skill schreiben
- [x] Datensatz (Synthetic-DGP + M4-Monatsdaten via API-Download)
- [x] Feature-Engineering (Lags, Rolling, Diffs, Kalender, exogene + Szenario-Treiber, Cross-Sectional)
- [x] Metriken (MAE, RMSE, sMAPE, Directional Accuracy, MASE in E5)
- [x] Graphen (reports/assets/, eingebettet in reports/report.html)
- [x] Experiments (Expanding-Window-Backtest; Studien E1-E5 in studies/)
