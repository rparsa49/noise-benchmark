import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load Data ----
json_path = Path("method_comparison_20250801-165359.json")
with open(json_path, "r") as f:
    data = json.load(f)

# ---- Helper to parse nested JSON strings ----
def parse_inner_json(entry):
    for key in ["clean", "noisy"]:
        entry[key] = json.loads(entry[key])
    return entry

# ---- Organize and parse data ----
results = defaultdict(lambda: defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))))
data = list(map(parse_inner_json, data))

for entry in data:
    phantom = entry["phantom"]
    thickness = float(entry["thickness"])
    kvp_key = f"{entry['kvp_pair'][0]}-{entry['kvp_pair'][1]}"
    for mode in ["clean", "noisy"]:
        payload = entry[mode]
        for i, mat in enumerate(payload["materials"]):
            results[kvp_key][thickness][mode][mat].append({
                "rho": payload["calculated_rhos"][i],
                "z": payload["calculated_z_effs"][i],
                "spr": payload["stopping_power"][i],
            })

# ---- Aggregate per material ----
summary_tables = defaultdict(lambda: defaultdict(dict))

for kvp_key in results:
    for thickness in results[kvp_key]:
        for mode in ["clean", "noisy"]:
            material_data = {}
            for mat, vals in results[kvp_key][thickness][mode].items():
                rho_avg = np.mean([v["rho"] for v in vals])
                z_avg = np.mean([v["z"] for v in vals])
                spr_avg = np.mean([v["spr"] for v in vals])
                material_data[mat] = {"avg_rho": rho_avg,
                                      "avg_z": z_avg, "avg_spr": spr_avg}
            df = pd.DataFrame(material_data).T
            df.index.name = "Material"
            summary_tables[kvp_key][mode][thickness] = df

# ---- Save CSV and plots ----
output_dir = Path("summary_outputs")
output_dir.mkdir(exist_ok=True)

for kvp_key in summary_tables:
    for mode in ["clean", "noisy"]:
        for thickness, df in summary_tables[kvp_key][mode].items():
            thickness_str = f"{thickness:.1f}".replace('.', '_')
            csv_name = f"{mode}_{kvp_key}_thickness_{thickness_str}.csv"
            df.to_csv(output_dir / csv_name)

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, col in zip(axes, ["avg_rho", "avg_z", "avg_spr"]):
                sns.barplot(x=df.index, y=df[col], ax=ax)
                ax.set_title(
                    f"{col} | {mode} | {kvp_key} | thickness {thickness}")
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha="right")
            plt.tight_layout()
            plot_name = f"{mode}_{kvp_key}_thickness_{thickness_str}.png"
            plt.savefig(output_dir / plot_name)
            plt.close()

print("Processing complete. Check the 'summary_outputs' folder for results.")
