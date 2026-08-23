"""Temporaerer Probe: sind alle vier Varianten-Prognosen belegt und plausibel?"""

import e6_levels_vs_logdiff as e6
import e7_gallery as g
import pandas as pd

raw = e6._panel(e6.REGIMES["stark_trendend"], seed=42, n_series=10)
s = sorted(raw["series"].unique())[0]
origin = raw["date"].max() - pd.DateOffset(months=18)
preds = g.variant_forecasts(raw, origin, [s])
truth = raw[(raw["series"] == s) & (raw["date"] > origin)]["value"].to_numpy()
print("series:", s, "| truth tail:", truth[-3:].round(1))
for m, df in preds.items():
    pp = df[df["series"] == s].sort_values("target_date")
    vals = pp["level_pred"].to_numpy()
    print(
        f"{m:18s} n={len(pp):3d} nan={pd.isna(vals).sum():3d} "
        f"first={vals[:3].round(1)} last={vals[-3:].round(1)}"
    )
