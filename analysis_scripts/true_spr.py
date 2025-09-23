import numpy as np
import json
from scipy.constants import physical_constants
from pathlib import Path

# Load Data from JSON
DATA_DIR = Path("./data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

MATERIAL_PROPERTIES = load_json("material_properties.json")
ELEMENTAL_PROPERTIES = load_json("element_properties.json")

TRUE_ZEFF = {mat: MATERIAL_PROPERTIES[mat]["Z_eff"]for mat in MATERIAL_PROPERTIES}
DENSITIES = {mat: MATERIAL_PROPERTIES[mat]["density"] for mat in MATERIAL_PROPERTIES}
MASSES = {ele: ELEMENTAL_PROPERTIES[ele]["mass"]for ele in ELEMENTAL_PROPERTIES}
COMPOSITIONS = {mat: MATERIAL_PROPERTIES[mat]["composition"] for mat in MATERIAL_PROPERTIES}

def spr(rho, I, beta):
    """
    rho : material mass density (from DENSITIES)
    I   : mean excitation energy (same units as Iw, see below)
    beta: v/c for the proton
    """
    me = 9.10938356e-31  
    c = 2.99792458e8   
    Iw = 75.3

    numerator = np.log(I / Iw)
    denominator = np.log((2 * me * (c**2) * (beta**2)) /
                         (Iw * (1 - (beta**2)))) - (beta**2)
    return rho * (1 - (numerator / denominator))

def beta(kvp):
    kinetic_energy_mev = kvp / 1000.0  
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) / proton_mass_mev
    return np.sqrt(1.0 - (1.0 / gamma**2))

def A(composition):
    total = 0.0
    for element, weight in composition.items():
        A_i = MASSES[element]
        total += weight / A_i
    return 1.0 / total

def I(Z):
    return 10.0 * Z


if __name__ == '__main__':
    KVPS = [70, 80, 100, 120, 140]
    out_dir = Path("validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    for kvp in KVPS:
        beta_val = beta(kvp)
        results = {}

        for mat, z in TRUE_ZEFF.items():
            rho_mat = DENSITIES[mat]
            I_mat = I(z)
            spr_val = spr(rho_mat, I_mat, beta_val)

            results[mat] = {
                "density": rho_mat,
                "atomic_number": z,
                "I_eV": I_mat,
                "beta": float(beta_val),
                "spr": float(spr_val)
            }

        output_path = out_dir / f"truth_{kvp}_KVP.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")
