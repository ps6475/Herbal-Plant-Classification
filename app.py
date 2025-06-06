import streamlit as st
import numpy as np
import json
from PIL import Image
import requests
import os

# --- Try to import TensorFlow safely ---
try:
    import tensorflow as tf
except ImportError:
    st.error("❌ TensorFlow failed to import. Make sure TensorFlow is installed and you're using Python 3.10 or compatible.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ TensorFlow error: {e}")
    st.stop()

# --- Constants ---
MODEL_PATH = "herbal_plant_classifier_v3_improved.h5"
CLASS_LABELS_PATH = "class_labels.json"
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Utilities ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_labels():
    with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_image(model, image, class_labels):
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    
    if str(class_idx) in class_labels:
        return class_labels[str(class_idx)], confidence
    else:
        return None, None

def generate_herb_explanation(herb_name):
    if not GROQ_API_KEY:
        return "🚫 Missing Groq API key. Please configure it."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert in herbal medicine and botany."},
            {"role": "user", "content": f"Provide the medicinal benefits, historical significance, and traditional uses of the herbal plant '{herb_name}'."}
        ]
    }
    
    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Groq API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"⚠️ API request failed: {e}"

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Herbal Plant Classifier", layout="centered")

    # Set background to white
    st.markdown("""
        <style>
            .stApp {
                background-color:white;
                color: black;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌿 Herbal Plant Classifier & Medicinal Uses (Groq AI)")

    model = load_model()
    class_labels = load_class_labels()

    uploaded_file = st.file_uploader("📤 Upload an image of an herbal plant", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        processed_image = preprocess_image(image)
        predicted_label, confidence = classify_image(model, processed_image, class_labels)

        if predicted_label:
            st.success(f"🌱 **Predicted Herb:** {predicted_label} (Confidence: {confidence:.2f})")

            with st.spinner("💡 Generating AI-based explanation..."):
                explanation = generate_herb_explanation(predicted_label)
                st.subheader("🤖 AI-Generated Explanation:")
                st.write(explanation)
        else:
            st.error("❌ Could not classify the image.")

if __name__ == "__main__":
    main()
