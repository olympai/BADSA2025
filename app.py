"""
Streamlit App for Skin Cancer Classification
Using trained MobileNetV2 model
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# Constants
IMG_SIZE = 224
MODEL_PATH = 'models/final_model.h5'
CLASS_NAMES_PATH = 'models/class_names.json'

@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first by running train_model.py")
        return None
    model = keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_class_names():
    """Load class names"""
    if not os.path.exists(CLASS_NAMES_PATH):
        # Default class names
        return {
            'akiec': 'Actinic keratoses',
            'bcc': 'Basal cell carcinoma',
            'bkl': 'Benign keratosis-like lesions',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic nevi',
            'vasc': 'Vascular lesions'
        }
    with open(CLASS_NAMES_PATH, 'r') as f:
        return json.load(f)

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict(model, image, class_names):
    """Make prediction on image"""
    # Preprocess
    processed_image = preprocess_image(image)

    # Predict
    predictions = model.predict(processed_image, verbose=0)[0]

    # Get sorted indices
    sorted_indices = np.argsort(predictions)[::-1]

    # Create results
    results = []
    class_keys = list(class_names.keys())
    for idx in sorted_indices:
        class_key = class_keys[idx]
        class_name = class_names[class_key]
        probability = predictions[idx]
        results.append({
            'class_key': class_key,
            'class_name': class_name,
            'probability': probability
        })

    return results

# App title and description
st.title("🔬 Hautkrebs-Klassifizierung")
st.markdown("""
Diese App verwendet ein **MobileNetV2-Modell mit Transfer Learning**, um Hautläsionen zu klassifizieren.
Das Modell wurde auf dem **HAM10000-Datensatz** trainiert und kann 7 verschiedene Arten von Hautläsionen erkennen.
""")

# Add info about classes
with st.expander("ℹ️ Informationen zu den Klassen"):
    st.markdown("""
    Das Modell kann folgende Hautläsionen klassifizieren:

    - **Actinic keratoses (akiec)**: Aktinische Keratosen - präkanzeröse Hautveränderungen
    - **Basal cell carcinoma (bcc)**: Basalzellkarzinom - häufigste Form von Hautkrebs
    - **Benign keratosis-like lesions (bkl)**: Gutartige keratoseartige Läsionen
    - **Dermatofibroma (df)**: Dermatofibrom - gutartige Hautveränderung
    - **Melanoma (mel)**: Melanom - bösartiger Hautkrebs
    - **Melanocytic nevi (nv)**: Melanozytäre Nävi - Muttermale
    - **Vascular lesions (vasc)**: Vaskuläre Läsionen - Gefäßveränderungen

    **⚠️ WICHTIG**: Diese App dient nur zu Demonstrations- und Bildungszwecken.
    Sie ersetzt KEINE professionelle medizinische Diagnose!
    """)

# Load model
model = load_model()
class_names = load_class_names()

if model is None:
    st.stop()

# File uploader
st.markdown("---")
st.header("📸 Bild hochladen")

uploaded_file = st.file_uploader(
    "Wähle ein Bild einer Hautläsion aus",
    type=['jpg', 'jpeg', 'png'],
    help="Unterstützte Formate: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Display image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hochgeladenes Bild")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        # Image info
        st.caption(f"Bildgröße: {image.size[0]} x {image.size[1]} Pixel")

    with col2:
        st.subheader("Klassifizierung")

        # Predict button
        if st.button("🔍 Klassifizieren", type="primary", use_container_width=True):
            with st.spinner("Analysiere Bild..."):
                # Make prediction
                results = predict(model, image, class_names)

                # Display top prediction
                top_result = results[0]
                confidence = top_result['probability'] * 100

                # Color based on confidence
                if confidence > 70:
                    color = "green"
                elif confidence > 50:
                    color = "orange"
                else:
                    color = "red"

                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;">
                    <h3 style="color: {color}; margin: 0;">Vorhersage: {top_result['class_name']}</h3>
                    <p style="font-size: 24px; margin: 10px 0;"><b>{confidence:.1f}%</b> Konfidenz</p>
                    <p style="color: #666; margin: 0;">Klasse: {top_result['class_key']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Display all predictions
                st.markdown("### Alle Vorhersagen")

                for result in results:
                    prob_percent = result['probability'] * 100
                    st.progress(result['probability'])
                    st.markdown(f"""
                    **{result['class_name']}** ({result['class_key']})
                    <span style="float: right;">{prob_percent:.2f}%</span>
                    """, unsafe_allow_html=True)
                    st.markdown("")

                # Warning message
                st.warning("""
                ⚠️ **WICHTIGER HINWEIS**:
                Diese Vorhersage dient nur zu Informationszwecken und ersetzt keine professionelle medizinische Diagnose.
                Bei Verdacht auf Hautkrebs oder anderen Hautveränderungen konsultieren Sie bitte einen Dermatologen!
                """)

else:
    st.info("👆 Bitte laden Sie ein Bild hoch, um mit der Klassifizierung zu beginnen.")

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ Über diese App")

    st.markdown("""
    ### Modell-Informationen
    - **Architektur**: MobileNetV2
    - **Methode**: Transfer Learning
    - **Datensatz**: HAM10000
    - **Anzahl Klassen**: 7
    - **Eingabegröße**: 224x224 Pixel

    ### Wie funktioniert es?
    1. Laden Sie ein Bild einer Hautläsion hoch
    2. Klicken Sie auf "Klassifizieren"
    3. Das Modell analysiert das Bild
    4. Sie erhalten eine Vorhersage mit Wahrscheinlichkeiten

    ### Technologie
    - TensorFlow/Keras
    - MobileNetV2 (ImageNet Pre-Training)
    - Streamlit

    ### HAM10000 Datensatz
    Der HAM10000 (Human Against Machine with 10000 training images)
    ist ein großer Datensatz dermatoskopischer Bilder verschiedener
    Hautläsionen, der für das Training von Machine Learning Modellen
    entwickelt wurde.
    """)

    st.markdown("---")
    st.markdown("Made with Streamlit & TensorFlow")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>Disclaimer</b>: Diese Anwendung dient ausschließlich zu Bildungs- und Demonstrationszwecken.
    Sie ist nicht für medizinische Diagnosen vorgesehen und sollte nicht als solche verwendet werden.
    Konsultieren Sie immer einen qualifizierten Arzt bei gesundheitlichen Bedenken.</p>
</div>
""", unsafe_allow_html=True)
