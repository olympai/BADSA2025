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

# Custom CSS for better design and readability
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Main content area */
    .main .block-container {
        background-color: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }

    /* Headers */
    h1, h2, h3 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }

    /* Improve button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
    }

    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #667eea !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] strong {
        color: white !important;
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Info/Warning boxes */
    .stAlert {
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

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
                <div style="padding: 25px; border-radius: 15px;
                     background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                     margin: 10px 0; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
                     border-left: 5px solid {color};">
                    <h3 style="color: #1a1a1a; margin: 0; font-size: 24px; font-weight: 700;">
                        Vorhersage: {top_result['class_name']}
                    </h3>
                    <p style="font-size: 36px; font-weight: 800; color: #1a1a1a; margin: 15px 0 10px 0;">
                        {confidence:.1f}%
                        <span style="font-size: 20px; font-weight: 600; color: #2d3748;">Konfidenz</span>
                    </p>
                    <p style="color: #2d3748; margin: 0; font-size: 16px; font-weight: 500;">
                        Klasse: <strong>{top_result['class_key']}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Display all predictions
                st.markdown("### Alle Vorhersagen")

                for result in results:
                    prob_percent = result['probability'] * 100
                    st.progress(float(result['probability']))
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #1a1a1a; font-size: 16px;">
                            {result['class_name']} <span style="color: #4a5568; font-size: 14px;">({result['class_key']})</span>
                        </span>
                        <span style="font-weight: 700; color: #2d3748; font-size: 16px;">{prob_percent:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

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
<div style="text-align: center; color: #2d3748; padding: 20px; background-color: #f7fafc; border-radius: 10px; margin-top: 20px;">
    <p style="font-size: 16px; line-height: 1.6; margin: 0;">
        <strong style="color: #1a1a1a; font-size: 18px;">Disclaimer</strong><br><br>
        <span style="color: #2d3748;">Diese Anwendung dient ausschließlich zu Bildungs- und Demonstrationszwecken.
        Sie ist nicht für medizinische Diagnosen vorgesehen und sollte nicht als solche verwendet werden.
        Konsultieren Sie immer einen qualifizierten Arzt bei gesundheitlichen Bedenken.</span>
    </p>
</div>
""", unsafe_allow_html=True)
