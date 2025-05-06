import pandas as pd
from mp_api.client import MPRester

# Replace with your actual API key
API_KEY = "YOUR_MP_API_KEY"

# Initialize MPRester
with MPRester(API_KEY) as mpr:
    # Query semiconductors with band gap between 0 and 5 eV
    docs = mpr.materials.summary.search(
        band_gap=(0, 5),  # Band gap range in eV
        fields=[
            "material_id", "formula_pretty", "band_gap", "density", "volume",
            "nsites", "formation_energy_per_atom", "energy_above_hull", "efermi",
            "is_metal", "is_gap_direct", "cbm", "vbm", "bulk_modulus", "shear_modulus",
            "symmetry", "is_stable", "decomposes_to", "e_total", "n", "bandstructure"
        ]
    )

# Convert to a properly formatted DataFrame
def flatten_data(doc):
    """Flatten the data to extract only relevant fields."""
    return {
        "Material ID": getattr(doc, "material_id", ""),
        "Formula": getattr(doc, "formula_pretty", ""),
        "Band Gap (eV)": getattr(doc, "band_gap", ""),
        "Density (g/cm³)": getattr(doc, "density", ""),
        "Volume (Å³)": getattr(doc, "volume", ""),
        "Number of Sites": getattr(doc, "nsites", ""),
        "Formation Energy per Atom (eV)": getattr(doc, "formation_energy_per_atom", ""),
        "Energy Above Hull (eV)": getattr(doc, "energy_above_hull", ""),
        "Fermi Energy (eV)": getattr(doc, "efermi", ""),
        "Is Metal": getattr(doc, "is_metal", ""),
        "Direct Band Gap": getattr(doc, "is_gap_direct", ""),
        "Conduction Band Minimum (eV)": getattr(doc, "cbm", ""),
        "Valence Band Maximum (eV)": getattr(doc, "vbm", ""),
        "Bulk Modulus (GPa)": getattr(doc.bulk_modulus, "vrh", "") if getattr(doc, "bulk_modulus", None) else "",
        "Shear Modulus (GPa)": getattr(doc.shear_modulus, "vrh", "") if getattr(doc, "shear_modulus", None) else "",

        # Available features from the search
        "Symmetry": getattr(doc, "symmetry", ""),
        "Is Stable": getattr(doc, "is_stable", ""),
        "Decomposes To": getattr(doc, "decomposes_to", ""),
        "Total Energy (eV)": getattr(doc, "e_total", ""),
        "Band Structure": getattr(doc, "bandstructure", "")
    }

# Apply flattening to all results
flattened_data = [flatten_data(doc) for doc in docs]

# Convert to DataFrame
df = pd.DataFrame(flattened_data)

# Display the first few rows
print(df.head())

# Save to CSV with proper headers
df.to_csv("semiconductor_data_with_available_features.csv", index=False)

