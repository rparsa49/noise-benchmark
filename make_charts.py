import csv
import re
from pathlib import Path
from collections import defaultdict
import math
import matplotlib.pyplot as plt

SELECTED_CSV = Path("point_summaries/selected_points.csv")
OUT_DIR = Path("plots_by_config_kvp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def infer_method_from_path(p: str) -> str:
    s = p.lower()
    for m in ("saito", "tanaka", "hunemohr", "schneider"):
        if m in s:
            return m.capitalize()
    return "Unknown"


def to_float(s):
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        try:
            return float(re.sub(r"[^0-9.+-eE]", "", str(s)))
        except Exception:
            return None


if not SELECTED_CSV.exists():
    raise FileNotFoundError(
        f"Missing {SELECTED_CSV}. Run the selection script first."
    )

rows = []
with SELECTED_CSV.open("r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        method = infer_method_from_path(row.get("selected_file", ""))
        thickness = str(row.get("thickness"))
        radius = to_float(row.get("radius"))
        kvp_pair = str(row.get("kvp_pair"))  
        noise = to_float(row.get("noise_level"))
        rmse_rho = row.get("rmse_rel_rho")
        rmse_z = row.get("rmse_rel_z")
        phantom_type = row.get("phantom")
        if None in (radius, noise) or rmse_rho is None or rmse_z is None:
            continue
        rows.append({
            "method": method,
            "thickness": thickness,
            "radius": radius,
            "kvp_pair": kvp_pair,
            "noise": noise,
            "rmse_rho": float(rmse_rho),
            "rmse_z": float(rmse_z),
            "phantom": phantom_type
        })

# Collapse duplicates across phantoms (if any):
# Keep a single row per (method, thickness, radius, kvp_pair, noise)
# using the minimal combined error.
best_by_noise = {}
for r in rows:
    key = (r["method"], r["thickness"], r["radius"], r["kvp_pair"], r["noise"], r["phantom"])
    combo = 0.5 * (r["rmse_rho"] + r["rmse_z"])
    if key not in best_by_noise or combo < best_by_noise[key]["combo"]:
        best_by_noise[key] = {"rmse_rho": r["rmse_rho"],
                              "rmse_z": r["rmse_z"], "combo": combo}

# Regroup for plotting: (method, thickness, radius, kvp_pair) -> list of items
groups = defaultdict(list)
for (method, thickness, radius, kvp, noise, phantom), vals in best_by_noise.items():
    groups[(method, thickness, radius, kvp, phantom)].append(
        {"noise": noise,
            "rmse_rho": vals["rmse_rho"], "rmse_z": vals["rmse_z"]}
    )

count = 0
for (method, thickness, radius, kvp_pair, phantom), items in groups.items():
    # sort by noise for nice x ordering
    items.sort(key=lambda d: d["noise"])

    x = [d["noise"] for d in items]
    y_rho = [d["rmse_rho"] for d in items]
    y_z = [d["rmse_z"] for d in items]

    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(x, y_rho, label="RMSE (ρ)", s=70, marker="o")
    plt.scatter(x, y_z,   label="RMSE (Z)", s=70, marker="o")

    plt.xlabel("Noise Level")
    plt.ylabel("RMSE (relative)")
    plt.title(
        f"{method}, slice thickness = {thickness}, radius = {radius}, kVp = {kvp_pair}, phantom = {phantom}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", title="Metric")
    plt.tight_layout()

    def safe(s): return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    out = OUT_DIR / \
        f"{safe(method)}_th{safe(thickness)}_r{safe(radius)}_kvp{safe(kvp_pair)}_phantom{safe(phantom)}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    count += 1

print(f"Saved {count} charts to {OUT_DIR.resolve()}")
