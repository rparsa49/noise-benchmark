# noise x st: do different slice thicknesses hold up better to noise?
# metric: which slice thickness had the lowest overall RMSE across all configurations (irrespective of model)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JSON_PATH = Path("/Users/royaparsa/Desktop/noise-benchmark/point_summaries/selected_points.json")

ST = [0.6, 1, 1.5, 2, 3, 4, 6, 8]
NOISE_TICKS = [0.01, 0.05, 0.1]
AGG_FUNC = "mean" 

_marker_cycle = ["o", "s", "^", "D", "P", "X", "v", ">"]
THICKNESS_MARKERS = {st: _marker_cycle[i % len(
    _marker_cycle)] for i, st in enumerate(ST)}

def parse_noise(val):
    if val is None:
        return np.nan
    s = str(val).strip().lower()
    if s in {"clean", "baseline", "none", ""}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_thickness(val):
    try:
        return float(val)
    except Exception:
        return np.nan

with JSON_PATH.open("r") as f:
    rows = json.load(f)

records = []
for r in rows:
    raw_noise = (r.get("noise_level", None))
    noise_val = parse_noise(raw_noise)

    st_val = parse_thickness(r.get("thickness"))

    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    records.append({
        "thickness": st_val,
        "noise_level": noise_val,
        "overall_rmse": overall,
    })

df = pd.DataFrame.from_records(records)

df = df[df["thickness"].isin(ST)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

agg = (df.groupby(["thickness", "noise_level"], as_index=False)["overall_rmse"]
         .agg(AGG_FUNC))

all_idx = pd.MultiIndex.from_product(
    [ST, NOISE_TICKS], names=["thickness", "noise_level"])
agg = (agg.set_index(["thickness", "noise_level"])
          .reindex(all_idx)
          .reset_index())

plt.figure(figsize=(8, 5))
for st in ST:
    sub = agg[agg["thickness"] == st]
    plt.scatter(
        sub["noise_level"], sub["overall_rmse"],
        marker=THICKNESS_MARKERS.get(st, "o"),
        label=f"{st} mm",
        s=70
    )

plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
plt.xlabel("Noise level")
plt.ylabel("Overall RMSE (mean of ρ and Z)")
plt.title("Noise vs Slice Thickness: Overall RMSE (lower is better)")
plt.grid(True, alpha=0.3)
plt.legend(title="Slice Thickness", frameon=False, ncol=2)
plt.tight_layout()
plt.show()
