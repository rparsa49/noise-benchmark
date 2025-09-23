# noise x phantom: which phantom size is more robust to noise
# metrics:
#   1) Overall RMSE (mean of rho & Z relative RMSEs)
#   2) SPR RMSE (relative)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JSON_PATH = Path(
    "/Users/royaparsa/Desktop/noise-benchmark/point_summaries_2/selected_point.json")

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


def aggregate_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    agg = (df.groupby(["size", "noise_level"], as_index=False)[metric_col]
             .agg(AGG_FUNC))
    all_idx = pd.MultiIndex.from_product(
        [SIZE_ORDER, NOISE_TICKS], names=["size", "noise_level"])
    agg = (agg.set_index(["size", "noise_level"])
              .reindex(all_idx)
              .reset_index())
    return agg


def scatter_by_size(agg: pd.DataFrame, metric_col: str, title: str, ylabel: str):
    plt.figure(figsize=(7, 4.5))
    for size in SIZE_ORDER:
        sub = agg[agg["size"] == size]
        plt.scatter(
            sub["noise_level"], sub[metric_col],
            marker=SIZE_MARKERS.get(size, "o"),
            label=size,
            s=70
        )
    plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
    plt.xlabel("Noise level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Phantom Size", frameon=False)
    plt.tight_layout()
    plt.show()


with JSON_PATH.open("r") as f:
    rows = json.load(f)

records = []
for r in rows:
    noise_val = parse_noise(r.get("noise_level", None))

    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    spr = float(r.get("rmse_rel_spr", np.nan))  # NEW

    records.append({
        "size": str(r.get("phantom", None)),
        "noise_level": noise_val,
        "overall_rmse": overall,
        "spr_rmse": spr,  # NEW
    })

df = pd.DataFrame.from_records(records)

df = df[df["size"].isin(SIZE_ORDER)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

# Plot 1: Overall (ρ+Z)
df_overall = df[np.isfinite(df["overall_rmse"])].copy()
if not df_overall.empty:
    agg_overall = aggregate_metric(df_overall, "overall_rmse")
    scatter_by_size(
        agg_overall,
        metric_col="overall_rmse",
        title="Noise Robustness: Overall RMSE by Phantom Size",
        ylabel="Overall RMSE (mean of ρ and Z, relative)"
    )
else:
    print("No overall_rmse data available to plot.")

# Plot 2: SPR RMSE
df_spr = df[np.isfinite(df["spr_rmse"])].copy()
if not df_spr.empty:
    agg_spr = aggregate_metric(df_spr, "spr_rmse")
    scatter_by_size(
        agg_spr,
        metric_col="spr_rmse",
        title="Noise Robustness: SPR RMSE by Phantom Size",
        ylabel="SPR RMSE (relative)"
    )
else:
    print("No spr_rmse data available to plot. Ensure 'rmse_rel_spr' is present in the JSON.")
