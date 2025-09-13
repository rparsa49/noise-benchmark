# noise x energy pair: which energy pair is more robust to noise?
# metric: which energy pair had the lowest overall RMSE across all configurations (irrespective of model)

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JSON_PATH = Path("/Users/royaparsa/Desktop/noise-benchmark/point_summaries/selected_points.json")

PAIR_ORDER = [(70, 100), (70, 120), (70, 140), (80, 100), (80, 120), (80, 140)]
NOISE_TICKS = [0.01, 0.05, 0.1]
AGG_FUNC = "mean"

_marker_cycle = ["o", "s", "^", "D", "P", "X"]
PAIR_MARKERS = {PAIR_ORDER[i]: _marker_cycle[i % len(_marker_cycle)] for i in range(len(PAIR_ORDER))}


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


def normalize_pair(p):
    if p is None:
        return None
    if isinstance(p, (list, tuple)) and len(p) == 2:
        try:
            return (int(p[0]), int(p[1]))
        except Exception:
            return None
    s = str(p)
    m = re.search(r"(\d+)\s*[/\-,]\s*(\d+)", s)
    if not m:
        m = re.search(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", s)
    if m:
        try:
            return (int(m.group(1)), int(m.group(2)))
        except Exception:
            return None
    return None


def parse_pair_from_any(row):

    kp = row.get("kvp_pair")
    pair = normalize_pair(kp)
    if pair:
        return pair
    return None

with JSON_PATH.open("r") as f:
    rows = json.load(f)

records = []
for r in rows:
    noise_val = parse_noise(r.get("noise_level", None))

    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    pair = parse_pair_from_any(r)

    records.append({
        "pair": pair,                
        "noise_level": noise_val,     
        "overall_rmse": overall,
    })

df = pd.DataFrame.from_records(records)

df = df[df["pair"].isin(PAIR_ORDER)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

agg = (df.groupby(["pair", "noise_level"], as_index=False)["overall_rmse"]
         .agg(AGG_FUNC))

all_idx = pd.MultiIndex.from_product(
    [PAIR_ORDER, NOISE_TICKS], names=["pair", "noise_level"])
agg = (agg.set_index(["pair", "noise_level"])
          .reindex(all_idx)
          .reset_index())

plt.figure(figsize=(8, 5))
for pair in PAIR_ORDER:
    sub = agg[agg["pair"] == pair]
    label = f"{pair[0]}/{pair[1]}"
    plt.scatter(
        sub["noise_level"], sub["overall_rmse"],
        marker=PAIR_MARKERS.get(pair, "o"),
        label=label,
        s=70
    )

plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
plt.xlabel("Noise level")
plt.ylabel("Overall RMSE (mean of ρ and Z)")
plt.title("Noise vs Energy Pair: Overall RMSE (lower is better)")
plt.grid(True, alpha=0.3)
plt.legend(title="Energy Pair (kVp)", frameon=False, ncol=2)
plt.tight_layout()
plt.show()
