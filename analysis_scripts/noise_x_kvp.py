# noise x energy pair: which energy pair is more robust to noise?
# metrics:
#   1) Overall RMSE (mean of rho & Z relative RMSEs)  [existing]
#   2) SPR RMSE (relative)                            [new]

import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JSON_PATH = Path(
    "/Users/royaparsa/Desktop/noise-benchmark/point_summaries_2/selected_point.json")

PAIR_ORDER = [(70, 100), (70, 120), (70, 140), (80, 100),
              (80, 120), (80, 140), (120, 120)]
NOISE_TICKS = [0.01, 0.05, 0.1]
AGG_FUNC = "mean"

_marker_cycle = ["o", "s", "^", "D", "P", "X"]
PAIR_MARKERS = {PAIR_ORDER[i]: _marker_cycle[i %
                                             len(_marker_cycle)] for i in range(len(PAIR_ORDER))}


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


def aggregate_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    agg = (df.groupby(["pair", "noise_level"], as_index=False)[metric_col]
             .agg(AGG_FUNC))
    all_idx = pd.MultiIndex.from_product(
        [PAIR_ORDER, NOISE_TICKS], names=["pair", "noise_level"])
    agg = (agg.set_index(["pair", "noise_level"])
              .reindex(all_idx)
              .reset_index())
    return agg


def scatter_by_pair(agg: pd.DataFrame, metric_col: str, title: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    for pair in PAIR_ORDER:
        sub = agg[agg["pair"] == pair]
        label = f"{pair[0]}/{pair[1]}"
        plt.scatter(
            sub["noise_level"], sub[metric_col],
            marker=PAIR_MARKERS.get(pair, "o"),
            label=label,
            s=70
        )
    plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
    plt.xlabel("Noise level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Energy Pair (kVp)", frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()


# ---------- Load ----------
with JSON_PATH.open("r") as f:
    rows = json.load(f)

records = []
for r in rows:
    noise_val = parse_noise(r.get("noise_level", None))
    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    # NEW: may be NaN if not present
    spr = float(r.get("rmse_rel_spr", np.nan))

    pair = parse_pair_from_any(r)

    records.append({
        "pair": pair,
        "noise_level": noise_val,
        "overall_rmse": overall,
        "spr_rmse": spr,   # NEW
    })

df = pd.DataFrame.from_records(records)

# Filter valid pairs & noise
df = df[df["pair"].isin(PAIR_ORDER)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

# -------- Plot 1: Overall (rho+Z) --------
df_overall = df[np.isfinite(df["overall_rmse"])].copy()
if not df_overall.empty:
    agg_overall = aggregate_metric(df_overall, "overall_rmse")
    scatter_by_pair(
        agg_overall,
        metric_col="overall_rmse",
        title="Noise vs Energy Pair: Overall RMSE (lower is better)",
        ylabel="Overall RMSE (mean of ρ and Z, relative)"
    )
else:
    print("No overall_rmse data available to plot.")

# -------- Plot 2: SPR RMSE (relative) --------
df_spr = df[np.isfinite(df["spr_rmse"])].copy()
if not df_spr.empty:
    agg_spr = aggregate_metric(df_spr, "spr_rmse")
    scatter_by_pair(
        agg_spr,
        metric_col="spr_rmse",
        title="Noise vs Energy Pair: SPR RMSE (lower is better)",
        ylabel="SPR RMSE (relative)"
    )
else:
    print("No spr_rmse data available to plot. Ensure 'rmse_rel_spr' is present in the JSON.")
