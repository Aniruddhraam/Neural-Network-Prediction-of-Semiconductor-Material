import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Symmetry options (must match what was used in training)
SYMMETRY_OPTIONS = [
    "Other", "P1", "P2", "P2/m", "C2", "C2h", "D2", "D2h", "C4", "C4h", "D4"
]

def preprocess_symmetry(symmetry_str):
    """Preprocess symmetry string into one-hot encoded list"""
    one_hot = [0] * len(SYMMETRY_OPTIONS)
    if symmetry_str in SYMMETRY_OPTIONS:
        one_hot[SYMMETRY_OPTIONS.index(symmetry_str)] = 1
    else:
        one_hot[0] = 1  # Default to "Other"
    return one_hot

def create_scaler():
    # Load your dataset
    df = pd.read_csv('semiconductor_data_with_available_features.csv')
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    
    # Fill missing values in numeric columns with their median values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Define the base feature columns used for prediction
    BASE_FEATURES = [
        "Density (g/cm³)",
        "Formation Energy per Atom (eV)",
        "Fermi Energy (eV)",
        "Is Metal",
        "Direct Band Gap",
        "Conduction Band Minimum (eV)",
        "Valence Band Maximum (eV)"
    ]
    
    # Extract the base feature columns
    X_df = df[BASE_FEATURES].copy()
    
    # Process symmetry for one-hot encoding
    symmetry_str = df["Symmetry"].astype(str)
    top_n = 10  # Must match training setting
    top_symmetry = symmetry_str.value_counts().nlargest(top_n).index
    symmetry_filtered = symmetry_str.apply(lambda x: x if x in top_symmetry else 'Other')
    
    # Create one-hot encoded symmetry features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    symmetry_one_hot = encoder.fit_transform(symmetry_filtered.values.reshape(-1, 1))
    symmetry_one_hot_df = pd.DataFrame(
        symmetry_one_hot, 
        columns=[f"Symmetry_{cat}" for cat in encoder.categories_[0]]
    )
    
    # Combine base features with symmetry features
    X_full = pd.concat([X_df, symmetry_one_hot_df], axis=1)
    
    # Standardize all features (including one-hot encoded)
    # Note: One-hot encoded features will be scaled but this is fine
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Full feature scaler (18 features) created and saved.")
    
    # Print feature statistics for reference
    print("\nFeature statistics after scaling:")
    scaled_df = pd.DataFrame(X_scaled, columns=X_full.columns)
    print(scaled_df.describe())
    
    print("\nOriginal feature statistics:")
    print(X_full.describe())

def prepare_features_for_prediction(features_dict, scaler_path='scaler.pkl'):
    """Prepare features for model prediction including one-hot encoding for symmetry."""
    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    # 1. Extract Base Features
    base_features_values = [
        features_dict.get("Density (g/cm³)", 0.0),
        features_dict.get("Formation Energy per Atom (eV)", 0.0),
        features_dict.get("Fermi Energy (eV)", 0.0),
        1.0 if features_dict.get("Is Metal", False) else 0.0,
        1.0 if features_dict.get("Direct Band Gap", False) else 0.0,
        features_dict.get("Conduction Band Minimum (eV)", 0.0),
        features_dict.get("Valence Band Maximum (eV)", 0.0)
    ]

    # 2. Preprocess Symmetry (get the 11 one-hot encoded values)
    symmetry_str = features_dict.get("Symmetry", "Other")  # Default to 'Other' if missing
    symmetry_one_hot_values = preprocess_symmetry(symmetry_str)
    # 3. Combine base features and symmetry features
    combined_features = base_features_values + symmetry_one_hot_values  # 7 + 11 = 18 features

    # Ensure we have exactly 18 features before scaling
    if len(combined_features) != 18:
        raise ValueError(f"Expected 18 features, but got {len(combined_features)}")

    # 4. Scale the combined features
    scaled_features = scaler.transform([combined_features])

    return scaled_features

if __name__ == "__main__":
    create_scaler()
