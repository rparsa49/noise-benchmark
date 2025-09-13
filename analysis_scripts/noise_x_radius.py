# noise x radius: do different radii hold up better to noise?
# metric: which radii had the lowest overall RMSE across all configurations (irrespective of model)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re 

JSON_PATH = Path("/Users/royaparsa/Desktop/noise-benchmark/point_summaries/selected_points.json")

RADII = [0.25, 0.5, 0.75, 1]
NOISE_TICKS = [0.01, 0.05, 0.1]
AGG_FUNC = "mean"

_marker_cycle = ["o", "s", "^", "D"]
RADII_MARKERS = {st: _marker_cycle[i % len(_marker_cycle)] for i, st in enumerate(RADII)}

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

def parse_radii(row):
    fpath = str(row.get("selected_file", "")).lower()
    m = re.search(r"comparison_([0-9]*\.?[0-9]+)", fpath)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return np.nan
    return np.nan

with JSON_PATH.open("r") as f:
    rows = json.load(f)

records = []
for r in rows:
    raw_noise = (r.get("noise_level", None))
    noise_val = parse_noise(raw_noise)

    radii_val = parse_radii(r)

    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    records.append({
        "radius": radii_val,
        "noise_level": noise_val,
        "overall_rmse": overall,
    })

df = pd.DataFrame.from_records(records)

df = df[df["radius"].isin(RADII)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

agg = (df.groupby(["radius", "noise_level"], as_index=False)["overall_rmse"]
         .agg(AGG_FUNC))

all_idx = pd.MultiIndex.from_product(
    [RADII, NOISE_TICKS], names=["radius", "noise_level"])
agg = (agg.set_index(["radius", "noise_level"])
          .reindex(all_idx)
          .reset_index())

plt.figure(figsize=(8, 5))
for radii in RADII:
    sub = agg[agg["radius"] == radii]
    plt.scatter(
        sub["noise_level"], sub["overall_rmse"],
        marker=RADII_MARKERS.get(radii, "o"),
        label=f"{radii}%",
        s=70
    )

plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
plt.xlabel("Noise level")
plt.ylabel("Overall RMSE (mean of ρ and Z)")
plt.title("Noise vs Radius: Overall RMSE (lower is better)")
plt.grid(True, alpha=0.3)
plt.legend(title="Radius", frameon=False, ncol=2)
plt.tight_layout()
plt.show()
