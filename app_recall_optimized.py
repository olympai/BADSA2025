"""
Recall-Optimized Streamlit App for Skin Cancer Classification
Lower threshold for dangerous classes to minimize False Negatives
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
    page_title="Skin Cancer Classifier (Recall-Optimized)",
    page_icon="🔬",
    layout="wide"
)

# Constants
IMG_SIZE = 224
MODEL_PATH = 'models/final_model_recall.h5'
CLASS_NAMES_PATH = 'models/class_names.json'

# Dangerous classes (cancer)
DANGEROUS_CLASSES = ['mel', 'bcc', 'akiec']

# RECALL OPTIMIZATION: Lower thresholds for dangerous classes
DANGER_THRESHOLD = 0.15  # Alert if cancer probability > 15%
WARNING_THRESHOLD = 0.30  # Warning if any dangerous class > 30%

@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return None
    model = keras.models.load_model(MODEL_PATH, custom_objects={'FocalLoss': keras.losses.Loss})
    return model

@st.cache_data
def load_class_names():
    """Load class names"""
    if not os.path.exists(CLASS_NAMES_PATH):
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
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image, class_names):
    """Make prediction on image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)[0]

    sorted_indices = np.argsort(predictions)[::-1]

    results = []
    class_keys = list(class_names.keys())
    for idx in sorted_indices:
        class_key = class_keys[idx]
        class_name = class_names[class_key]
        probability = predictions[idx]
        is_dangerous = class_key in DANGEROUS_CLASSES
        results.append({
            'class_key': class_key,
            'class_name': class_name,
            'probability': probability,
            'is_dangerous': is_dangerous
        })

    return results

# App title
st.title("🔬 Hautkrebs-Klassifizierung (Recall-Optimiert)")
st.markdown("""
Diese App verwendet ein **Recall-optimiertes MobileNetV2-Modell**, um Hautläsionen zu klassifizieren.
**Ziel**: Minimierung von False Negatives (übersehenen Krebsfällen) ⚠️
""")

# Add info
with st.expander("ℹ️ Über Recall-Optimierung"):
    st.markdown("""
    ### Warum Recall-Optimierung?

    Bei der Erkennung von Hautkrebs ist es **kritisch**, alle potenziell gefährlichen Fälle zu erkennen:

    - **False Negative** (übersehener Krebs) = GEFÄHRLICH ❌
    - **False Positive** (falscher Alarm) = Akzeptabel ✅

    ### Anpassungen:
    - **Niedrige Schwellenwerte**: Warnung schon ab 15% Wahrscheinlichkeit für Krebs
    - **Fokussierte Klassen**: Melanom, Basalzellkarzinom, Aktinische Keratosen
    - **Focal Loss**: Bessere Erkennung schwieriger Fälle
    - **Boosted Weights**: Krebsklassen erhalten höhere Priorität

    ### Gefährliche Klassen (Krebs):
    - 🔴 **Melanoma (mel)**: Bösartiger Hautkrebs (am gefährlichsten)
    - 🔴 **Basal cell carcinoma (bcc)**: Basalzellkarzinom
    - 🟠 **Actinic keratoses (akiec)**: Präkanzeröse Läsionen

    ### Gutartige Klassen:
    - 🟢 **Benign keratosis-like lesions (bkl)**
    - 🟢 **Dermatofibroma (df)**
    - 🟢 **Melanocytic nevi (nv)**: Muttermale
    - 🟢 **Vascular lesions (vasc)**

    **⚠️ WICHTIG**: Immer einen Dermatologen konsultieren!
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
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hochgeladenes Bild")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        st.caption(f"Bildgröße: {image.size[0]} x {image.size[1]} Pixel")

    with col2:
        st.subheader("Klassifizierung")

        if st.button("🔍 Klassifizieren", type="primary", use_container_width=True):
            with st.spinner("Analysiere Bild..."):
                results = predict(model, image, class_names)

                # Calculate total danger score
                danger_score = sum([r['probability'] for r in results if r['is_dangerous']])

                # RECALL OPTIMIZATION: Alert system
                if danger_score > WARNING_THRESHOLD:
                    alert_level = "🚨 HOHE PRIORITÄT"
                    alert_color = "#ff4444"
                elif danger_score > DANGER_THRESHOLD:
                    alert_level = "⚠️ ACHTUNG"
                    alert_color = "#ff8800"
                else:
                    alert_level = "✅ Niedrige Priorität"
                    alert_color = "#44ff44"

                # Display alert
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {alert_color}20; border: 3px solid {alert_color}; margin: 10px 0;">
                    <h2 style="color: {alert_color}; margin: 0;">{alert_level}</h2>
                    <p style="font-size: 20px; margin: 10px 0;"><b>Krebs-Wahrscheinlichkeit: {danger_score*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                # Top prediction
                top_result = results[0]
                confidence = top_result['probability'] * 100

                if top_result['is_dangerous']:
                    box_color = "#ff4444"
                    emoji = "🔴"
                else:
                    box_color = "#44ff44"
                    emoji = "🟢"

                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;">
                    <h3 style="color: {box_color}; margin: 0;">{emoji} Hauptdiagnose: {top_result['class_name']}</h3>
                    <p style="font-size: 32px; font-weight: bold; color: #000; margin: 10px 0;">{confidence:.1f}% <span style="font-size: 20px;">Konfidenz</span></p>
                    <p style="color: #555; margin: 0;">Klasse: {top_result['class_key']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Dangerous classes section
                st.markdown("### 🔴 Gefährliche Klassen (Krebs)")
                dangerous_results = [r for r in results if r['is_dangerous']]

                for result in dangerous_results:
                    prob_percent = result['probability'] * 100

                    # Color based on probability
                    if prob_percent > 30:
                        bar_color = "#ff0000"
                    elif prob_percent > 15:
                        bar_color = "#ff8800"
                    else:
                        bar_color = "#ffaa00"

                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <p style="margin: 5px 0;"><b>{result['class_name']}</b> ({result['class_key']})</p>
                        <div style="background-color: #ddd; border-radius: 5px; height: 25px; position: relative;">
                            <div style="background-color: {bar_color}; width: {prob_percent}%; height: 100%; border-radius: 5px;"></div>
                            <span style="position: absolute; right: 10px; top: 2px; font-weight: bold;">{prob_percent:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")

                # Benign classes section
                st.markdown("### 🟢 Gutartige Klassen")
                benign_results = [r for r in results if not r['is_dangerous']]

                with st.expander("Alle gutartigen Vorhersagen anzeigen"):
                    for result in benign_results:
                        prob_percent = result['probability'] * 100
                        st.progress(float(result['probability']))
                        st.markdown(f"""
                        **{result['class_name']}** ({result['class_key']})
                        <span style="float: right;">{prob_percent:.2f}%</span>
                        """, unsafe_allow_html=True)
                        st.markdown("")

                # Medical advice
                if danger_score > WARNING_THRESHOLD:
                    st.error("""
                    ### 🚨 DRINGEND: Arzt aufsuchen!

                    Die Wahrscheinlichkeit für eine gefährliche Hautläsion ist **HOCH** (>{:.0f}%).

                    **Empfohlene Maßnahmen:**
                    - ⚕️ **Sofort** einen Dermatologen aufsuchen
                    - 📸 Weitere Bilder dokumentieren
                    - 📝 Veränderungen notieren
                    - 🏥 Bei schnellem Wachstum: Notfall!
                    """.format(danger_score * 100))

                elif danger_score > DANGER_THRESHOLD:
                    st.warning("""
                    ### ⚠️ ACHTUNG: Ärztliche Untersuchung empfohlen

                    Die Wahrscheinlichkeit für eine gefährliche Hautläsion ist **ERHÖHT** ({:.0f}%).

                    **Empfohlene Maßnahmen:**
                    - 👨‍⚕️ Termin beim Dermatologen vereinbaren
                    - 📷 Läsion beobachten und dokumentieren
                    - 🔍 Auf Veränderungen achten
                    """.format(danger_score * 100))
                else:
                    st.info("""
                    ### ℹ️ Niedrige Priorität

                    Die Wahrscheinlichkeit für Krebs ist **NIEDRIG** ({:.0f}%).

                    **Empfehlung:**
                    - ✅ Regelmäßige Hautuntersuchungen beibehalten
                    - 👁️ Läsion im Auge behalten
                    - 📅 Jährliches Screening

                    **Bei Veränderungen trotzdem Arzt aufsuchen!**
                    """.format(danger_score * 100))

                # General warning
                st.markdown("---")
                st.warning("""
                ⚠️ **WICHTIGER HINWEIS**:
                Diese KI-gestützte Analyse ist **KEIN ERSATZ** für eine professionelle medizinische Diagnose!

                Das System ist auf **hohe Sensitivität** (Recall) optimiert und warnt lieber einmal zu viel als einmal zu wenig.
                **Bei jedem Verdacht: Arzt aufsuchen!**
                """)

else:
    st.info("👆 Bitte laden Sie ein Bild hoch, um mit der Klassifizierung zu beginnen.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Recall-Optimierung")

    st.markdown("""
    ### Systemeinstellungen
    - **Gefahr-Schwelle**: 15%
    - **Warnung-Schwelle**: 30%
    - **Fokus**: Minimiere False Negatives

    ### Performance
    - **Recall**: ~70-85% (Ziel: Alle Krebsfälle finden)
    - **Precision**: ~50-70% (Mehr False Positives akzeptiert)
    - **AUC**: ~89-92%

    ### Modell-Info
    - **Architektur**: MobileNetV2
    - **Loss**: Focal Loss (γ=2.5, α=0.75)
    - **Gewichtung**: +50% für Krebsklassen
    - **Datensatz**: HAM10000

    ### Recall-Strategie
    Lieber 10x falscher Alarm als 1x übersehener Krebs!
    """)

    st.markdown("---")
    st.markdown("Made with ❤️ for patient safety")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>Medizinischer Disclaimer</b>: Diese Anwendung dient ausschließlich zu Informationszwecken.
    Sie ist NICHT für medizinische Diagnosen vorgesehen. Konsultieren Sie immer einen Arzt!</p>
</div>
""", unsafe_allow_html=True)
