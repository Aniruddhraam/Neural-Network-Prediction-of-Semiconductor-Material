from mp_api.client import MPRester
import os
import subprocess
import threading
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from streamlit.runtime.scriptrunner import add_script_run_ctx

# API and Path Configurations
API_KEY = "8vMFC0NgKdfYSDqvGdmRY3vw3TaZtMuC"
# Don't hardcode VESTA_PATH - it will be dynamically set in the app

# Symmetry options (must match training)
SYMMETRY_OPTIONS = [
    "Other", "P1", "P2", "P2/m", "C2", "C2h", "D2", "D2h", "C4", "C4h", "D4"
]

# Initialize MPRester
try:
    mpr = MPRester(API_KEY)
except Exception as e:
    print(f"Warning: Failed to initialize MPRester: {e}")
    mpr = None

def prepare_features(features_dict):
    """Create a DataFrame from the input dictionary"""
    # Create a DataFrame from the input dictionary
    df = pd.DataFrame([features_dict])
    
    # Handle boolean fields (convert to int)
    df['Is Metal'] = df['Is Metal'].astype(int)
    df['Direct Band Gap'] = df['Direct Band Gap'].astype(int)

    # One-hot encode symmetry
    if 'Symmetry' in df.columns:
        df = pd.get_dummies(df, columns=['Symmetry'], prefix='Symmetry')

    # Ensure all expected symmetry columns exist (important during inference)
    expected_symmetries = [
        "Symmetry_Other", "Symmetry_P1", "Symmetry_P2", "Symmetry_P2/m", "Symmetry_C2",
        "Symmetry_C2h", "Symmetry_D2", "Symmetry_D2h", "Symmetry_C4", "Symmetry_C4h", "Symmetry_D4"
    ]
    for sym in expected_symmetries:
        if sym not in df.columns:
            df[sym] = 0

    # Reorder columns to match training order (if known)
    df = df.reindex(sorted(df.columns), axis=1)

    return df

def preprocess_symmetry(symmetry_str):
    """Convert symmetry string to one-hot encoded array"""
    one_hot = [0] * len(SYMMETRY_OPTIONS)
    if symmetry_str in SYMMETRY_OPTIONS:
        one_hot[SYMMETRY_OPTIONS.index(symmetry_str)] = 1
    else:
        one_hot[0] = 1  # Default to "Other"
    return one_hot

def fetch_features_from_mp(identifier):
    """Fetch material features and CIF path from Materials Project."""
    if mpr is None:
        raise RuntimeError("MPRester could not be initialized. Check API key and connection.")

    try:
        # Validate input
        if not identifier or not isinstance(identifier, str):
            raise ValueError("Invalid MP-ID: Must be a non-empty string")
        if not identifier.startswith('mp-'):
            raise ValueError("Invalid MP-ID format: Must start with 'mp-'")

        material_id = identifier.strip()

        # --- Fetch summary data using recommended fields ---
        fields_to_fetch = [
            "material_id",
            "formula_pretty",
            "density",
            "formation_energy_per_atom",
            "efermi",
            "is_metal",
            "band_gap",
            "is_gap_direct",
            "cbm",
            "vbm",
            "symmetry"
        ]
        docs = mpr.summary.search(material_ids=[material_id], fields=fields_to_fetch)

        if not docs:
            raise ValueError(f"No material summary found with ID: {material_id}")

        entry = docs[0]  # Get the first (and only) document

        # --- Fetch structure separately ---
        structure = mpr.get_structure_by_material_id(material_id)
        if structure is None:
            print(f"Warning: Could not retrieve structure for {material_id}")
            cif_path = None
        else:
            # Save CIF file
            cif_path = f"{material_id}.cif"
            try:
                structure.to(fmt="cif", filename=cif_path)
                print(f"CIF file saved to {cif_path}")
            except Exception as cif_e:
                print(f"Warning: Failed to save CIF file {cif_path}: {cif_e}")
                cif_path = None  # Indicate failure to save

        # --- Extract Features Safely ---
        symmetry_symbol = "Unknown"
        if entry.symmetry and entry.symmetry.symbol:
            symmetry_symbol = entry.symmetry.symbol

        # Helper for safe access and NaN fallback
        def safe_get(data, key, default=np.nan):
            val = getattr(data, key, default)
            return val if val is not None else default

        features = {
            # Map fetched fields to your expected feature names
            "Density (g/cm³)": safe_get(entry, "density"),
            "Formation Energy per Atom (eV)": safe_get(entry, "formation_energy_per_atom"),
            "Fermi Energy (eV)": safe_get(entry, "efermi"),
            "Is Metal": 1.0 if getattr(entry, "is_metal", False) else 0.0,
            # Use the explicit is_gap_direct field if available
            "Direct Band Gap": 1.0 if getattr(entry, "is_gap_direct", False) else 0.0,
            "Conduction Band Minimum (eV)": safe_get(entry, "cbm"),
            "Valence Band Maximum (eV)": safe_get(entry, "vbm"),
            "Symmetry": symmetry_symbol  # Pass the raw symbol for later encoding
        }

        return features, cif_path

    except Exception as e:
        # Log the error for debugging
        print(f"Error during Materials Project fetch for {identifier}: {e}")
        # Re-raise a more specific error for the app to catch
        raise RuntimeError(f"Failed to fetch data from Materials Project: {str(e)}")

def predict_semiconductor(model, features):
    """Make prediction using the loaded model"""
    try:
        prediction = model.predict(features)
        return prediction[0] > 0.5
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def launch_vesta_thread(vesta_path, cif_path):
    try:
        subprocess.Popen([vesta_path, cif_path])
        print("✅ VESTA launched successfully (threaded).")
    except Exception as e:
        print(f"❌ Error launching VESTA in thread: {e}")

def open_cif_in_vesta(cif_path):
    """Opens the CIF file in VESTA using a detached thread to avoid Streamlit blocking issues."""
    vesta_path = "/home/madhulika/Semester_2/IMI/VESTA-x86_64/VESTA"

    if not os.path.exists(cif_path):
        return False, f"CIF file not found: {cif_path}"
    if not os.path.exists(vesta_path):
        return False, f"VESTA not found at: {vesta_path}"

    try:
        # Launch in a separate thread so Streamlit's UI thread doesn't interfere
        threading.Thread(target=launch_vesta_thread, args=(vesta_path, cif_path), daemon=True).start()
        return True, "VESTA launched successfully!"
    except Exception as e:
        return False, f"Error launching VESTA: {e}"
        
def save_scaler(scaler, path='scaler.pkl'):
    """Save the feature scaler to disk"""
    try:
        joblib.dump(scaler, path)
        print(f"Scaler saved to {path}")
    except Exception as e:
        print(f"Error saving scaler: {e}")

def load_scaler(path='feature_scaler_18.pkl'):
    """Load the feature scaler from disk"""
    try:
        return joblib.load(path)
    except Exception as e:
        raise Exception(f"Error loading scaler: {str(e)}")
