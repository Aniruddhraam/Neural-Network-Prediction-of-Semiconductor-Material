import streamlit as st
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model
import json

# ------------------------------------------------------------------
# 1) Load your Keras model & scaler
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model  = load_model("material_classifier_3way.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ------------------------------------------------------------------
# 2) Talk to Ollama via OpenAI‐compatible API
# ------------------------------------------------------------------
def ask_ollama(prompt: str, model_name: str = "llama2", max_tokens: int = 1024) -> str:
    """
    Sends a chat request to Ollama's OpenAI‐compatible endpoint and returns the assistant's full reply.
    """
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a materials‐science expert."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    r = requests.post(url, json=payload)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        # On unexpected response, return raw JSON for debugging
        return json.dumps(data, indent=2)

# ------------------------------------------------------------------
# 3) Streamlit UI
# ------------------------------------------------------------------
st.title("🔬 Material Classifier + LLM Insights")
st.write("Predict Metal / Semiconductor / Insulator and get AI-powered explanations.")

# User inputs
bg    = st.number_input("Band Gap (eV)", format="%.4f")
dens  = st.number_input("Density (g/cm³)", format="%.4f")
feap  = st.number_input("Formation Energy per Atom (eV)", format="%.4f")
fermi = st.number_input("Fermi Energy (eV)", format="%.4f")

if st.button("Predict & Explain"):
    # Predict
    X     = np.array([[bg, dens, feap, fermi]])
    Xs    = scaler.transform(X).reshape(-1, 4, 1)
    probs = model.predict(Xs)[0]
    idx   = int(np.argmax(probs))
    labels = ["Metal", "Semiconductor", "Insulator"]

    st.subheader(f"🔮 Predicted class: **{labels[idx]}**")
    st.write("Class probabilities:")
    for lbl, p in zip(labels, probs):
        st.write(f"- {lbl}: {p:.3f}")

    # Build prompt with word limit
    prob_dict = {lbl: round(float(p), 3) for lbl, p in zip(labels, probs)}
    prompt = (
        f"I have a material with these properties:\n"
        f"- Band Gap: {bg} eV\n"
        f"- Density: {dens} g/cm³\n"
        f"- Formation Energy per Atom: {feap} eV\n"
        f"- Fermi Energy: {fermi} eV\n\n"
        f"The classifier predicted **{labels[idx]}** with probabilities {prob_dict}.\n"
        f"Explain why the model might have chosen this class and suggest two next experimental steps."
        f" Please limit your response to no more than 200 words."
    )

    with st.spinner("🧠 Generating LLM insights…"):
        try:
            explanation = ask_ollama(prompt)
            st.markdown("**LLM Explanation & Next Steps:**")
            st.write(explanation)
        except Exception as e:
            st.error(f"Could not reach Ollama: {e}")

# ------------------------------------------------------------------
# 4) Freeform LLM Q&A
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("🗣️ Ask the LLM about materials science")
user_q = st.text_area("Your question", "")
if st.button("Ask Ollama", key="qa_button"):
    if user_q.strip():
        with st.spinner("🧠 Thinking…"):
            try:
                # Add word limit here as well if desired
                ans = ask_ollama(user_q)
                st.write(ans)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question first.")

