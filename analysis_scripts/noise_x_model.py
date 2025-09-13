# noise x model: which models are more robust to noise
# metric: which models had the lowest overall RMSE across all configurations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

JSON_PATH = Path("/Users/royaparsa/Desktop/noise-benchmark/point_summaries/selected_points.json")

MODEL_MARKERS = {"saito": "o", "tanaka": "s", "hunemohr": "^"}
MODEL_ORDER = ["saito", "tanaka", "hunemohr"]
NOISE_TICKS = [0.01, 0.05, 0.1]  
AGG_FUNC = "mean"             


def infer_model(selected_file: str) -> str:
    f = (selected_file or "").lower()
    if "saito" in f:
        return "saito"
    if "tanaka" in f:
        return "tanaka"
    if "hunemohr" in f:
        return "hunemohr"


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
    model = infer_model(r.get("selected_file", ""))
    raw_noise = (r.get("noise_level", None))
    noise_val = parse_noise(raw_noise)

    rho = float(r.get("rmse_rel_rho", np.nan))
    zee = float(r.get("rmse_rel_z", np.nan))
    overall = np.nanmean([rho, zee])

    records.append({
        "model": model,
        "noise_level": noise_val,
        "overall_rmse": overall,
    })

df = pd.DataFrame.from_records(records)

df = df[df["model"].isin(MODEL_ORDER)].copy()
df = df[np.isfinite(df["noise_level"])].copy()

agg = (df.groupby(["model", "noise_level"], as_index=False)["overall_rmse"]
         .agg(AGG_FUNC))

all_idx = pd.MultiIndex.from_product(
    [MODEL_ORDER, NOISE_TICKS], names=["model", "noise_level"])
agg = (agg.set_index(["model", "noise_level"])
          .reindex(all_idx)
          .reset_index())

plt.figure(figsize=(7, 4.5))
for model in MODEL_ORDER:
    sub = agg[agg["model"] == model]
    plt.scatter(
        sub["noise_level"], sub["overall_rmse"],
        marker=MODEL_MARKERS.get(model, "o"),
        label=model.capitalize(),
        s=70
    )

plt.xticks(NOISE_TICKS, [str(x) for x in NOISE_TICKS])
plt.xlabel("Noise level")
plt.ylabel("Overall RMSE (mean of ρ and Z)")
plt.title("Noise Robustness: Overall RMSE by Model")
plt.grid(True, alpha=0.3)
plt.legend(title="Method", frameon=False)
plt.tight_layout()
plt.show()