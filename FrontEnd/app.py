import streamlit as st
# Assuming utils.py has fetch_features_from_mp and open_cif_in_vesta
from utils import fetch_features_from_mp, open_cif_in_vesta
import tensorflow as tf
import joblib
import os
import numpy as np
import pandas as pd
import traceback # Import traceback for detailed errors

# --- Page Config MUST BE THE FIRST Streamlit command ---
st.set_page_config(layout="wide")
# ------------------------------------------------------

# --- Configuration matching MLP.py ---
EXPECTED_NUMERICAL_FEATURES = [
    "Density (g/cm³)", "Formation Energy per Atom (eV)", "Fermi Energy (eV)",
    "Is Metal", "Direct Band Gap", "Conduction Band Minimum (eV)", "Valence Band Maximum (eV)"
]
# --- End Configuration ---

# --- Load Medians (Optional but Recommended) ---
# Display loading status/errors in the sidebar *after* page config
median_values = None
try:
    median_values = joblib.load('median_values.pkl')
    st.sidebar.info("Loaded pre-calculated median values for imputation.")
except FileNotFoundError:
    st.sidebar.warning("Median values file ('median_values.pkl') not found. Using 0 for imputation (less accurate).")
    median_values = pd.Series(0.0, index=EXPECTED_NUMERICAL_FEATURES)
except Exception as e:
    st.sidebar.error(f"Error loading median values: {e}. Using 0 for imputation.")
    median_values = pd.Series(0.0, index=EXPECTED_NUMERICAL_FEATURES)
# --- End Load Medians ---


# Initialize session state (before loading models, as it doesn't use st commands)
if 'cif_path' not in st.session_state:
    st.session_state.cif_path = None
if 'open_vesta' not in st.session_state:
    st.session_state.open_vesta = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_features' not in st.session_state:
    st.session_state.last_features = None
if 'last_symmetry' not in st.session_state:
    st.session_state.last_symmetry = None


# --- Load Model, Scaler, Encoder ---
# Use sidebar for status, handle critical errors
model = None
scaler = None
encoder = None
try:
    model = tf.keras.models.load_model('semiconductor_classifier_tf32_batch256.keras')
    st.sidebar.success("✅ Model Loaded")
    scaler = joblib.load('scaler.pkl')
    st.sidebar.success("✅ Scaler Loaded")
    encoder = joblib.load('symmetry_encoder.pkl')
    st.sidebar.success("✅ Encoder Loaded")
except Exception as e:
    # Display error prominently and stop if essential components fail
    st.error(f"CRITICAL ERROR: Failed to load model/scaler/encoder: {str(e)}")
    st.error(f"Traceback: {traceback.format_exc()}")
    st.sidebar.error("❌ Loading Failed")
    st.stop() # Stop execution if core components fail
# --- End Load Model etc ---


# --- Function Definitions ---

def display_feature_table(features_dict, symmetry=None):
    """Display features in a nice table format using st.metric"""
    st.subheader("Material Features Used for Prediction")
    col1, col2 = st.columns(2)

    def format_value(key, format_str="{:.4f}", is_bool=False):
        value = features_dict.get(key)
        if value is None or pd.isna(value):
            return "N/A"
        if is_bool:
            return "Yes" if bool(value) else "No"
        try:
            return format_str.format(float(value))
        except (ValueError, TypeError):
            return str(value)

    with col1:
        st.metric(label="Density (g/cm³)", value=format_value("Density (g/cm³)"))
        st.metric(label="Formation Energy (eV/atom)", value=format_value("Formation Energy per Atom (eV)"))
        st.metric(label="Fermi Energy (eV)", value=format_value("Fermi Energy (eV)"))
        st.metric(label="Is Metal", value=format_value("Is Metal", is_bool=True))

    with col2:
        st.metric(label="Direct Band Gap", value=format_value("Direct Band Gap", is_bool=True))
        st.metric(label="CBM (eV)", value=format_value("Conduction Band Minimum (eV)"))
        st.metric(label="VBM (eV)", value=format_value("Valence Band Maximum (eV)"))
        if symmetry:
            st.metric(label="Symmetry (Input)", value=symmetry if symmetry else "N/A")


def prepare_and_scale_features(features_dict, raw_symmetry_str, scaler_obj, encoder_obj, median_vals_series):
    """
    Prepares numerical/symmetry features, imputes missing numericals,
    ensures order, and scales. Returns scaled NumPy array or raises error.
    """
    # Check if scaler/encoder are loaded
    if scaler_obj is None or encoder_obj is None:
        raise RuntimeError("Scaler or Encoder objects are not loaded. Cannot prepare features.")
    if median_vals_series is None:
         # This case should ideally be handled by the fallback during loading
         st.warning("Median values Series not available, using 0 for all imputations.")
         median_vals_series = pd.Series(0.0, index=EXPECTED_NUMERICAL_FEATURES)


    # 1. Prepare Numerical Features
    numerical_data = {}
    missing_features = []
    invalid_features = {}
    for feature_name in EXPECTED_NUMERICAL_FEATURES:
        value = features_dict.get(feature_name)
        if value is None or pd.isna(value):
            missing_features.append(feature_name)
            numerical_data[feature_name] = np.nan
        elif isinstance(value, bool):
             numerical_data[feature_name] = int(value)
        else:
             try: numerical_data[feature_name] = float(value)
             except (ValueError, TypeError):
                  invalid_features[feature_name] = value
                  numerical_data[feature_name] = np.nan # Treat invalid as missing

    if missing_features: st.warning(f"Missing: {', '.join(missing_features)}. Imputing.")
    if invalid_features:
        msgs = [f"'{k}' (value: {v})" for k, v in invalid_features.items()]
        st.warning(f"Invalid values: {', '.join(msgs)}. Treating as missing.")

    numerical_features_df = pd.DataFrame([numerical_data], columns=EXPECTED_NUMERICAL_FEATURES)

    # 2. Impute Numerical NaNs
    try:
        impute_values = median_vals_series.reindex(numerical_features_df.columns)
        numerical_features_df.fillna(impute_values, inplace=True)
        if numerical_features_df.isnull().values.any(): # Check post-imputation
             st.warning("NaNs remain after median imputation, filling with 0.")
             numerical_features_df.fillna(0, inplace=True)
    except Exception as impute_err:
         st.error(f"Error during imputation: {impute_err}. Falling back to fillna(0).")
         numerical_features_df.fillna(0, inplace=True)

    # 3. Encode Symmetry
    try:
        symmetry_input_df = pd.DataFrame({"Symmetry": [raw_symmetry_str if raw_symmetry_str else "Unknown"]})
        symmetry_encoded_array = encoder_obj.transform(symmetry_input_df)
        if hasattr(symmetry_encoded_array, "toarray"): symmetry_encoded_array = symmetry_encoded_array.toarray()
        symmetry_feature_names = encoder_obj.get_feature_names_out(["Symmetry"])
        symmetry_encoded_df = pd.DataFrame(symmetry_encoded_array, columns=symmetry_feature_names)
    except Exception as e:
         raise ValueError(f"Failed to encode symmetry '{raw_symmetry_str}': {e}. Cannot proceed.")

    # 4. Concatenate
    features_combined = pd.concat([numerical_features_df.reset_index(drop=True),
                                   symmetry_encoded_df.reset_index(drop=True)], axis=1)

    # 5. Final Checks & Scaling
    if features_combined.isnull().values.any():
         st.error("FATAL: NaNs detected before scaling. Check imputation/encoding.")
         features_combined.fillna(0, inplace=True) # Last resort

    expected_count = getattr(scaler_obj, 'n_features_in_', None)
    if expected_count is not None and features_combined.shape[1] != expected_count:
         raise ValueError(f"Feature count mismatch: Got {features_combined.shape[1]}, expected {expected_count}.")

    try:
        scaled_features = scaler_obj.transform(features_combined.astype("float32"))
        if np.isnan(scaled_features).any():
            st.warning("NaNs detected *after* scaling (potential zero variance?). Replacing with 0.")
            scaled_features = np.nan_to_num(scaled_features, nan=0.0)
    except Exception as e:
         raise RuntimeError(f"Error during scaling: {e}")

    return scaled_features


def predict_semiconductor(model_obj, scaled_features_array):
    """Predicts using the model. Returns (bool, float) or (None, None)."""
    if model_obj is None:
         raise RuntimeError("Model not loaded. Cannot predict.")
    if scaled_features_array is None or not isinstance(scaled_features_array, np.ndarray):
         raise ValueError("Invalid input to predict_semiconductor.")
    if np.isnan(scaled_features_array).any():
        st.warning("Input to prediction contains NaN. Output may be unreliable.")
        # scaled_features_array = np.nan_to_num(scaled_features_array, nan=0.0) # Optional: Impute here too

    try:
        prediction_proba = model_obj.predict(scaled_features_array)
        if prediction_proba is None or np.isnan(prediction_proba).any():
            st.error("Model prediction resulted in NaN.")
            return None, None
        probability = float(prediction_proba[0][0])
        prediction = (probability > 0.5)
        return bool(prediction), probability
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None


# --- Main App Logic ---

def main():
    # Page config is now done globally, before this function runs

    st.title("Semiconductor Material Classifier")
    st.markdown("Predict semiconductor properties via Materials Project ID or manual input.")

    # Check if essential components loaded correctly before proceeding
    if model is None or scaler is None or encoder is None or median_values is None:
         st.error("Application cannot start because essential components (model, scaler, encoder, or medians) failed to load. Check sidebar and logs.")
         st.stop()

    st.sidebar.title("Status") # Add title to sidebar section

    tab1, tab2 = st.tabs(["Materials Project Lookup", "Manual Input"])

    with tab1:
        st.header("Fetch from Materials Project")
        st.markdown("Enter an MP-ID (e.g., `mp-149`, `mp-2534`).")

        col_input, col_button = st.columns([3, 1])
        with col_input: mpid_input = st.text_input("MP-ID", key="mpid_input", placeholder="mp-XXXXX", label_visibility="collapsed").strip()
        with col_button: fetch_clicked = st.button("Fetch & Predict", key="fetch_button", type="primary", use_container_width=True)

        # Placeholders for results to appear below the button
        results_placeholder_mp = st.container()

        if fetch_clicked:
            if not mpid_input: results_placeholder_mp.error("Please enter an MP-ID."); st.stop()
            if not mpid_input.lower().startswith('mp-'): results_placeholder_mp.error("Invalid MP-ID format."); st.stop()

            with results_placeholder_mp: # Display progress and results within this container
                with st.spinner(f"Processing {mpid_input}..."):
                    try:
                        # Fetch
                        features_dict, cif_path = fetch_features_from_mp(mpid_input)
                        if not features_dict: raise ValueError(f"No data retrieved for {mpid_input}. Check ID/API key.")

                        symmetry_raw = features_dict.get("Symmetry", "Unknown")
                        st.session_state.last_symmetry = symmetry_raw
                        st.session_state.last_features = features_dict

                        # Prepare & Scale
                        scaled_features = prepare_and_scale_features(features_dict, symmetry_raw, scaler, encoder, median_values)

                        # Predict
                        prediction, probability = predict_semiconductor(model, scaled_features)

                        # Display Results
                        if prediction is None: raise RuntimeError("Prediction failed, cannot display results.")

                        st.subheader(f"Results for {mpid_input}")
                        pred_col1, pred_col2 = st.columns([1, 2])
                        with pred_col1:
                            pred_text = ":green[Semiconductor]" if prediction else ":red[Non-Semiconductor]"
                            conf = probability if prediction else 1 - probability
                            st.markdown(f"#### Prediction: {pred_text}")
                            if pd.isna(conf): st.warning("Confidence score NaN.")
                            elif 0.0 <= conf <= 1.0: st.progress(conf); st.caption(f"Confidence: {conf:.2%}")
                            else: st.warning(f"Conf. score ({conf:.4f}) out of range.")

                             # CIF/VESTA buttons
                            st.markdown("---") # Separator
                            st.subheader("Structure")
                            if cif_path and os.path.exists(cif_path):
                                 full_cif_path = os.path.abspath(cif_path)
                                 st.session_state.cif_path = full_cif_path
                                 cif_dl_col, vesta_col = st.columns(2)
                                 try:
                                      with open(full_cif_path, "rb") as f:
                                           cif_dl_col.download_button("Download CIF", f, file_name=f"{mpid_input}.cif", mime="chemical/x-cif")
                                 except Exception as file_e: st.warning(f"CIF access error: {file_e}")
                            else: st.info("CIF file not available."); st.session_state.cif_path = None

                        with pred_col2:
                            display_feature_table(features_dict, symmetry_raw)


                    except (ValueError, RuntimeError, ImportError) as e: st.error(f"Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Unexpected Error: {str(e)}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        
        # Button to open in VESTA (sets a flag in session state)
        if st.session_state.cif_path and st.button("Open in VESTA", key="vesta_button"):
            st.session_state.open_vesta = True
            st.rerun()

    # Trigger VESTA opening after rerun
    if st.session_state.open_vesta:
        st.session_state.open_vesta = False  # Reset the trigger
        success, message = open_cif_in_vesta(st.session_state.cif_path)
        if success:
            st.success(message)
        else:
            st.error(message)                    

    with tab2:
        st.header("Manual Feature Input")
        st.markdown("Enter properties manually.")

        try:
            known_symmetries = ["Other"] + encoder.categories_[0].tolist()
            known_symmetries = sorted(list(set(known_symmetries)))
        except Exception: known_symmetries = ["Other", "P1", "Fm-3m"] # Fallback

        with st.form("manual_input_form"):
            # Form layout (as before)
            st.markdown("**Numerical Properties:**")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1: density = st.number_input("Density (g/cm³)", 0.0, value=2.33, format="%.4f", key="man_density"); is_metal = st.checkbox("Is Metal?", key="man_is_metal")
            with mcol2: formation_energy = st.number_input("Form. Energy (eV/atom)", value=0.0, format="%.4f", key="man_form_e"); direct_band_gap = st.checkbox("Direct Gap?", key="man_direct_gap")
            with mcol3: fermi_energy = st.number_input("Fermi Energy (eV)", value=0.0, format="%.4f", key="man_fermi_e")

            st.markdown("**Band Structure:**")
            bcol1, bcol2 = st.columns(2)
            with bcol1: cbm = st.number_input("CBM (eV)", value=0.0, format="%.4f", key="man_cbm")
            with bcol2: vbm = st.number_input("VBM (eV)", value=0.0, format="%.4f", key="man_vbm")

            st.markdown("**Symmetry:**")
            symmetry_input = st.selectbox("Space Group", known_symmetries, index=0, key="man_symmetry")

            submitted = st.form_submit_button("Predict Manual Input", type="primary", use_container_width=True)

            # Placeholder for manual results inside the form
            results_placeholder_manual = st.empty()

            if submitted:
                 with results_placeholder_manual.container(): # Display results here
                     with st.spinner("Processing manual input..."):
                          features_dict = {
                              "Density (g/cm³)": density, "Formation Energy per Atom (eV)": formation_energy,
                              "Fermi Energy (eV)": fermi_energy, "Is Metal": is_metal,
                              "Direct Band Gap": direct_band_gap, "Conduction Band Minimum (eV)": cbm,
                              "Valence Band Maximum (eV)": vbm
                          }
                          symmetry_raw = symmetry_input
                          st.session_state.last_symmetry = symmetry_raw
                          st.session_state.last_features = features_dict
                          st.session_state.cif_path = None

                          try:
                              scaled_features = prepare_and_scale_features(features_dict, symmetry_raw, scaler, encoder, median_values)
                              prediction, probability = predict_semiconductor(model, scaled_features)

                              if prediction is None: raise RuntimeError("Prediction failed.")

                              st.subheader("Manual Input Prediction")
                              res_col1_man, res_col2_man = st.columns(2)
                              with res_col1_man:
                                   pred_text = ":green[Semiconductor]" if prediction else ":red[Non-Semiconductor]"
                                   conf = probability if prediction else 1 - probability
                                   st.markdown(f"#### Prediction: {pred_text}")
                                   if pd.isna(conf): st.warning("Confidence NaN.")
                                   elif 0.0 <= conf <= 1.0: st.progress(conf); st.caption(f"Conf: {conf:.2%}")
                                   else: st.warning(f"Conf ({conf:.4f}) out of range.")
                              with res_col2_man:
                                   display_feature_table(features_dict, symmetry_raw)

                          except (ValueError, RuntimeError) as e: st.error(f"Error: {str(e)}")
                          except Exception as e:
                              st.error(f"Unexpected Error: {str(e)}")
                              st.error(f"Traceback: {traceback.format_exc()}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure essential components are loaded before calling main
    if model is None or scaler is None or encoder is None or median_values is None:
         st.error("App initialization failed: Essential components could not be loaded.")
         # Optionally log the error here if running in a container/server
    else:
         main()
