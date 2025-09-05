import json
import math
from pathlib import Path
from collections import defaultdict
import re
import csv

TRUTH_PATH = Path("true_mats.json")

# Base folder that contains many ...comparison_*.json files (Saito/Tanaka/Hunemohr etc.)
RESULTS_DIR = Path("results")    
GLOB_PATTERN = "**/*comparison_*.json" 


BRANCH = "noisy"  

# selection metric: combine rho/Z relative-error RMSEs
def combined_metric(rmse_rho_rel, rmse_z_rel):
    return 0.5 * (rmse_rho_rel + rmse_z_rel)

def load_truth(path: Path):
    with path.open("r") as f:
        truth = json.load(f)

    true_rho = {}
    true_z = {}
    rho_key = "rho_e_w"
    z_key = "Z_eff"
    for mat, vals in truth.items():
        true_rho[mat] = vals[rho_key]
        true_z[mat] = vals[z_key]
    return true_rho, true_z

def rmse_relative(preds, truths):
    """
    preds/truths are lists (same length).
    Uses relative error: |pred - truth| / |truth|
    Returns sqrt(mean(rel_err^2))
    """
    rel_sq = []
    for p, t in zip(preds, truths):
        if t == 0 or t is None:

            continue
        rel = abs(p - t) / abs(t)
        rel_sq.append(rel * rel)
    if not rel_sq:
        return float("nan")
    return math.sqrt(sum(rel_sq) / len(rel_sq))


def safe_parse_branch(obj):
    """
    Input row['clean'] or row['noisy'] is a JSON string. Parse and return dict.
    """
    if isinstance(obj, str):
        return json.loads(obj)
    return obj 


def extract_radius(filename: str):
    m = re.search(r"_comparison_([0-9.]+)_", filename)
    return float(m.group(1)) if m else None

true_rho, true_z = load_truth(TRUTH_PATH)

# Group rows by (phantom, thickness, kvp_pair, noise, radius)
groups = defaultdict(list)

all_files = sorted(RESULTS_DIR.glob(GLOB_PATTERN))
if not all_files:
    print(f"No files matched under {RESULTS_DIR}/{GLOB_PATTERN}.")


for file in all_files:
    try:
        with file.open("r") as f:
            rows = json.load(f)
    except Exception as e:
        print(f"Skipping {file}: {e}")
        continue

    radius = extract_radius(file.name)

    for row in rows:
        phantom = row.get("phantom")
        thickness = str(row.get("thickness"))
        kvp_pair = tuple(row.get("kvp_pair", []))  # (low, high)
        noise = str(row.get("noise_level"))
        pair_index = row.get("pair_index")

        branch = safe_parse_branch(row.get(BRANCH))
        mats = branch.get("materials", [])
        rhos = branch.get("calculated_rhos", [])
        zs = branch.get("calculated_z_effs", [])

        # Align predictions and truths for materials we have truth for
        preds_rho, truths_rho = [], []
        preds_z, truths_z = [], []

        for m, pr, pz in zip(mats, rhos, zs):
            if m in true_rho and m in true_z:
                preds_rho.append(pr)
                truths_rho.append(true_rho[m])
                preds_z.append(pz)
                truths_z.append(true_z[m])

        # Compute per-image RMSE of relative errors
        rmse_rho_rel = rmse_relative(preds_rho, truths_rho)
        rmse_z_rel = rmse_relative(preds_z, truths_z)

        key = (phantom, thickness, kvp_pair, noise, radius)
        groups[key].append({
            "file": str(file),
            "pair_index": pair_index,
            "rmse_rho_rel": rmse_rho_rel,
            "rmse_z_rel": rmse_z_rel,
            "combined": combined_metric(rmse_rho_rel, rmse_z_rel),
            "n_materials": len(preds_rho)
        })


selected = []
for key, items in groups.items():
    # Keep only items with finite metrics
    finite = [it for it in items if math.isfinite(it["combined"])]
    if not finite:
        continue
    best = min(finite, key=lambda it: it["combined"])
    phantom, thickness, kvp_pair, noise, radius = key
    selected.append({
        "phantom": phantom,
        "thickness": thickness,
        "kvp_pair": f"{kvp_pair[0]}-{kvp_pair[1]}" if len(kvp_pair) == 2 else str(kvp_pair),
        "noise_level": noise,
        "radius": radius,
        "pair_index": best["pair_index"],
        "rmse_rel_rho": best["rmse_rho_rel"],   
        "rmse_rel_z": best["rmse_z_rel"],       
        "selected_file": best["file"],
        "n_materials": best["n_materials"]
    })


OUT_DIR = Path("point_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)
json_out = OUT_DIR / "selected_points.json"
csv_out = OUT_DIR / "selected_points.csv"

with json_out.open("w") as f:
    json.dump(selected, f, indent=2)

with csv_out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "phantom", "thickness", "kvp_pair", "noise_level", "radius",
        "pair_index", "rmse_rel_rho", "rmse_rel_z", "selected_file", "n_materials"
    ])
    writer.writeheader()
    writer.writerows(selected)
