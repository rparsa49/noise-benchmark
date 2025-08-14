import numpy as np
import json
import scipy.constants
from scipy.constants import physical_constants
from pathlib import Path

# Reference: https://en.wikipedia.org/wiki/Bethe_formula

# Load Data from JSON
DATA_DIR = Path("data")

def load_json(file_name):
    with open(DATA_DIR / file_name, "r") as file:
        return json.load(file)

MATERIAL_PROPERTIES = load_json("material_properties.json")
ELEMENTAL_PROPERTIES = load_json("element_properties.json")
TRUE_ZEFF = {mat: MATERIAL_PROPERTIES[mat]["Z_eff"]for mat in MATERIAL_PROPERTIES}
DENSITIES = {mat: MATERIAL_PROPERTIES[mat]["density"] for mat in MATERIAL_PROPERTIES}
MASSES = {ele: ELEMENTAL_PROPERTIES[ele]["mass"] for ele in ELEMENTAL_PROPERTIES}
COMPOSITIONS = {mat: MATERIAL_PROPERTIES[mat]["composition"] for mat in MATERIAL_PROPERTIES}

# Bethe Bloch Equation
def spr(N, I, beta):
    '''
    N = elecontron density
    I = mean excitation energy
    z = charge 
    Beta = v/c
    '''
    me = 9.10938356e-31
    e = 1.60217663e-19
    c = 2.99792458e8
    z = physical_constants['elementary charge'][0]

    first_term = (4 * np.pi) / (me * (c ** 2))
    second_term = (N * (z ** 2) / beta)
    third_term = (e**2) / (4 * np.pi * scipy.constants.epsilon_0)
    fourth_term = np.log((2.0 * me * c**2 * (beta ** 2)) /(I * (1.0 - (beta ** 2)))) - (beta ** 2)

    return first_term * second_term * third_term * fourth_term
    
# Beta Proton
def beta(kvp):
    kinetic_energy_mev = kvp / 1000
    proton_mass_mev = physical_constants['proton mass energy equivalent in MeV'][0]
    gamma = (proton_mass_mev + kinetic_energy_mev) / proton_mass_mev
    return np.sqrt(1 - (1 / gamma ** 2))

# Electron Density
def n(rho, Z, A):
    '''
    rho = density of material
    Z = effective atomic number
    A = relative mass
    '''
    return (scipy.constants.Avogadro * Z * rho) / (A * physical_constants['molar mass constant'][0])

# Relative Atomic Weight
def A(composition):
    total = 0.0
    for element, weight in composition.items():
        A_i = MASSES[element]
        total += weight / A_i
    
    return 1 / total

# Mean Excitation Energy
def I(Z):
    return 10 * Z

if __name__ == '__main__':
    KVPS = [70, 80, 100, 120, 140]
    for kvp in KVPS:
        
        Is = {}
        ns = {}
        sprs = {}
    
        for mat, z in TRUE_ZEFF.items():
            # Obtain true mean excitation energy for each
            Is[mat] = I(z)
    
            # Calculate electron density for each material
            a_i = A(COMPOSITIONS[mat])
            ns[mat] = n(DENSITIES[mat], z, a_i) / 1e26

            # Calculate beta proton
            beta2 = beta(kvp)
            
            # Calculate true SPR
            sprs[mat] = spr(ns[mat], Is[mat], beta2)
        
        results = {}
        for mat in ns.keys():
            results[mat] = {
                'electron density': ns.get(mat),
                "atomic number": TRUE_ZEFF[mat],
                "spr": sprs[mat]
            }
        output_path = Path(f"validation/truth_{kvp}_KVP.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")

    