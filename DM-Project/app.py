"""
app.py  –  CoralCare Streamlit Web Application
───────────────────────────────────────────────
Upload a coral image → get a health-status prediction.

Pre-requisites (run these FIRST):
    python step1_download_dataset.py
    python step2_preprocess.py
    python step3_feature_extraction.py
    python step4_train_models.py

Then launch:
    streamlit run app.py
"""

import cv2
import numpy as np
import joblib
import streamlit as st

from config import MODEL_PATH, SCALER_PATH, ENCODER_PATH


# ─── Load saved artefacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    le       = joblib.load(ENCODER_PATH)
    return rf_model, scaler, le


rf_model, scaler, le = load_artifacts()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoralCare – Coral Bleaching Classifier",
    page_icon="🪸",
    layout="centered",
)

st.title("🌊 CoralCare – Coral Bleaching Classification")
st.write(
    "Upload a coral image to predict its health status: "
    "**bleached**, **unbleached**, or **dead**."
)

# ─── File uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    # Decode uploaded bytes to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Uploaded Image", use_column_width=True)

    # ── Feature extraction ───────────────────────────────────────────────────
    img_resized = cv2.resize(img, (224, 224))
    avg_color   = img_resized.mean(axis=(0, 1))   # BGR
    brightness  = float(img_resized.mean())
    texture     = float(img_resized.std())
    coral_count = 1   # Assume at least one coral in the uploaded image

    features = np.array([[
        avg_color[0], avg_color[1], avg_color[2],
        brightness, texture, coral_count
    ]])

    # Show extracted features
    st.subheader("📊 Extracted Features")
    st.json({
        "avg_B":        round(float(avg_color[0]), 2),
        "avg_G":        round(float(avg_color[1]), 2),
        "avg_R":        round(float(avg_color[2]), 2),
        "brightness":   round(brightness, 2),
        "texture":      round(texture, 2),
        "coral_count":  coral_count,
    })

    # ── Prediction ───────────────────────────────────────────────────────────
    features_scaled = scaler.transform(features)
    pred_encoded    = rf_model.predict(features_scaled)
    pred_label      = le.inverse_transform(pred_encoded)[0]
    pred_proba      = rf_model.predict_proba(features_scaled)[0]

    # Display probabilities
    st.subheader("🔍 Prediction Probabilities")
    prob_df = {cls: round(float(p), 3) for cls, p in zip(le.classes_, pred_proba)}
    st.bar_chart(prob_df)

    # Final result
    st.subheader("🪸 Coral Health Status")
    if pred_label == "bleached":
        st.warning(f"⚠️  **BLEACHED** – The coral shows signs of bleaching.")
    elif pred_label == "dead":
        st.error(f"☠️  **DEAD** – The coral appears to be dead.")
    else:
        st.success(f"✅  **HEALTHY (Unbleached)** – The coral looks healthy!")
