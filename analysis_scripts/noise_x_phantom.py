# noise x phantom: which phantom size is more robust to noise
# metric: which phantom size had the lowest overall RMSE across all configurations irregardless of model

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JSON_PATH = Path("/Users/royaparsa/Desktop/noise-benchmark/point_summaries/selected_points.json")

SIZE_MARKERS = {"Body-Abdomen": "o", "Head-Abdomen": "s"}
SIZE_ORDER = ["Body-Abdomen", "Head-Abdomen"]
NOISE_TICKS = [0.01, 0.05, 0.1]
AGG_FUNC = "mean"

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


with JSON_PATH.open("r") as f:
    rows = json.load(f)

records = []
for r in rows:
    raw_noise = (r.get("noise_level", None))
    noise_val = parse_noise(raw_noise)

    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    records.append({
        "size": str(r.get("phantom", None)),
        "noise_level": noise_val,
        "overall_rmse": overall,
    })

df = pd.DataFrame.from_records(records)

df = df[df["size"].isin(SIZE_ORDER)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

agg = (df.groupby(["size", "noise_level"], as_index=False)["overall_rmse"]
         .agg(AGG_FUNC))

all_idx = pd.MultiIndex.from_product(
    [SIZE_ORDER, NOISE_TICKS], names=["size", "noise_level"])
agg = (agg.set_index(["size", "noise_level"])
          .reindex(all_idx)
          .reset_index())

plt.figure(figsize=(7, 4.5))
for size in SIZE_ORDER:
    sub = agg[agg["size"] == size]
    plt.scatter(
        sub["noise_level"], sub["overall_rmse"],
        marker=SIZE_MARKERS.get(size, "o"),
        label=size.capitalize(),
        s=70
    )

plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
plt.xlabel("Noise level")
plt.ylabel("Overall RMSE (mean of ρ and Z)")
plt.title("Noise Robustness: Overall RMSE by Phantom Size")
plt.grid(True, alpha=0.3)
plt.legend(title="Phantom Size", frameon=False)
plt.tight_layout()
plt.show()
