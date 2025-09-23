# noise x st: do different slice thicknesses hold up better to noise?
# metrics:
#   1) Overall RMSE (mean of rho & Z relative RMSEs)  [existing]
#   2) SPR RMSE (relative)                            [new]

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JSON_PATH = Path(
    "/Users/royaparsa/Desktop/noise-benchmark/point_summaries_2/selected_point.json")

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


def aggregate_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    agg = (df.groupby(["thickness", "noise_level"], as_index=False)[metric_col]
             .agg(AGG_FUNC))
    all_idx = pd.MultiIndex.from_product(
        [ST, NOISE_TICKS], names=["thickness", "noise_level"])
    agg = (agg.set_index(["thickness", "noise_level"])
              .reindex(all_idx)
              .reset_index())
    return agg


def scatter_by_thickness(agg: pd.DataFrame, metric_col: str, title: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    for st in ST:
        sub = agg[agg["thickness"] == st]
        plt.scatter(
            sub["noise_level"], sub[metric_col],
            marker=THICKNESS_MARKERS.get(st, "o"),
            label=f"{st} mm",
            s=70
        )
    plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
    plt.xlabel("Noise level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Slice Thickness", frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()


# -------- Load & prepare --------
with JSON_PATH.open("r") as f:
    rows = json.load(f)

records = []
for r in rows:
    noise_val = parse_noise(r.get("noise_level", None))
    st_val = parse_thickness(r.get("thickness"))

    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    spr = float(r.get("rmse_rel_spr", np.nan))  # NEW

    records.append({
        "thickness": st_val,
        "noise_level": noise_val,
        "overall_rmse": overall,
        "spr_rmse": spr,  # NEW
    })

df = pd.DataFrame.from_records(records)

# Filter valid slice thicknesses & noise
df = df[df["thickness"].isin(ST)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

# -------- Plot 1: Overall (rho+Z) --------
df_overall = df[np.isfinite(df["overall_rmse"])].copy()
if not df_overall.empty:
    agg_overall = aggregate_metric(df_overall, "overall_rmse")
    scatter_by_thickness(
        agg_overall,
        metric_col="overall_rmse",
        title="Noise vs Slice Thickness: Overall RMSE (lower is better)",
        ylabel="Overall RMSE (mean of ρ and Z, relative)"
    )
else:
    print("No overall_rmse data available to plot.")

# -------- Plot 2: SPR RMSE (relative) --------
df_spr = df[np.isfinite(df["spr_rmse"])].copy()
if not df_spr.empty:
    agg_spr = aggregate_metric(df_spr, "spr_rmse")
    scatter_by_thickness(
        agg_spr,
        metric_col="spr_rmse",
        title="Noise vs Slice Thickness: SPR RMSE (lower is better)",
        ylabel="SPR RMSE (relative)"
    )
else:
    print("No spr_rmse data available to plot. Ensure 'rmse_rel_spr' is present in the JSON.")
