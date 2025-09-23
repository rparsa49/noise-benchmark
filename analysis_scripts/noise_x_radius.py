# noise x radius: do different radii hold up better to noise?
# metrics:
#   1) Overall RMSE (mean of rho & Z relative RMSEs)
#   2) SPR RMSE (relative)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

JSON_PATH = Path(
    "/Users/royaparsa/Desktop/noise-benchmark/point_summaries_2/selected_point.json")

RADII = [0.25, 0.5, 0.75, 1]
NOISE_TICKS = [0.01, 0.05, 0.1]
AGG_FUNC = "mean"

_marker_cycle = ["o", "s", "^", "D"]
RADII_MARKERS = {st: _marker_cycle[i % len(
    _marker_cycle)] for i, st in enumerate(RADII)}


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
    # from filename pattern "...comparison_<radius>..."
    fpath = str(row.get("selected_file", "")).lower()
    m = re.search(r"comparison_([0-9]*\.?[0-9]+)", fpath)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return np.nan
    return np.nan


def aggregate_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    agg = (df.groupby(["radius", "noise_level"], as_index=False)[metric_col]
             .agg(AGG_FUNC))
    all_idx = pd.MultiIndex.from_product(
        [RADII, NOISE_TICKS], names=["radius", "noise_level"])
    agg = (agg.set_index(["radius", "noise_level"])
              .reindex(all_idx)
              .reset_index())
    return agg


def scatter_by_radius(agg: pd.DataFrame, metric_col: str, title: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    for r in RADII:
        sub = agg[agg["radius"] == r]
        plt.scatter(
            sub["noise_level"], sub[metric_col],
            marker=RADII_MARKERS.get(r, "o"),
            label=f"{r}",
            s=70
        )
    plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
    plt.xlabel("Noise level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Radius", frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()


with JSON_PATH.open("r") as f:
    rows = json.load(f)

records = []
for r in rows:
    noise_val = parse_noise(r.get("noise_level", None))
    radii_val = parse_radii(r)

    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    spr = float(r.get("rmse_rel_spr", np.nan))  # NEW

    records.append({
        "radius": radii_val,
        "noise_level": noise_val,
        "overall_rmse": overall,
        "spr_rmse": spr,  # NEW
    })

df = pd.DataFrame.from_records(records)

df = df[df["radius"].isin(RADII)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

# Plot 1: Overall (ρ+Z)
df_overall = df[np.isfinite(df["overall_rmse"])].copy()
if not df_overall.empty:
    agg_overall = aggregate_metric(df_overall, "overall_rmse")
    scatter_by_radius(
        agg_overall,
        metric_col="overall_rmse",
        title="Noise vs Radius: Overall RMSE (lower is better)",
        ylabel="Overall RMSE (mean of ρ and Z, relative)"
    )
else:
    print("No overall_rmse data available to plot.")

# Plot 2: SPR RMSE
df_spr = df[np.isfinite(df["spr_rmse"])].copy()
if not df_spr.empty:
    agg_spr = aggregate_metric(df_spr, "spr_rmse")
    scatter_by_radius(
        agg_spr,
        metric_col="spr_rmse",
        title="Noise vs Radius: SPR RMSE (lower is better)",
        ylabel="SPR RMSE (relative)"
    )
else:
    print("No spr_rmse data available to plot. Ensure 'rmse_rel_spr' is present in the JSON.")
