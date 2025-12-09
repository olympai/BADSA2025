"""
BADSA 2025 Group 3
Streamlit App for Skin Cancer Classification
Using trained MobileNetV2 model
Enhanced with advanced features
"""

# Here we initiate the libraries we need
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# Constants
IMG_SIZE = 224
MODEL_PATH = 'models/final_model.h5' # This is the directory we used to save our best trained model
CLASS_NAMES_PATH = 'models/class_names.json' # This is the directory for the class names

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

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

@st.cache_data
def get_condition_details():
    """Get detailed information about each condition"""
    return {
        'akiec': {
            'name': 'Actinic Keratoses',
            'description': 'Aktinische Keratosen sind raue, schuppige Flecken auf der Haut, die sich durch jahrelange Sonneneinstrahlung entwickeln.',
            'risk_level': 'Mittel',
            'malignant': False,
            'precancerous': True,
            'prevalence': 'Häufig (ca. 15-20% der Erwachsenen)',
            'treatment': 'Kryotherapie, topische Medikamente, photodynamische Therapie',
            'emoji': '⚠️'
        },
        'bcc': {
            'name': 'Basal Cell Carcinoma',
            'description': 'Das Basalzellkarzinom ist die häufigste Form von Hautkrebs. Es wächst langsam und metastasiert selten.',
            'risk_level': 'Mittel-Hoch',
            'malignant': True,
            'precancerous': False,
            'prevalence': 'Sehr häufig (ca. 80% aller Hautkrebsfälle)',
            'treatment': 'Chirurgische Entfernung, Mohs-Chirurgie, Strahlentherapie',
            'emoji': '🔴'
        },
        'bkl': {
            'name': 'Benign Keratosis-like Lesions',
            'description': 'Gutartige keratoseartige Läsionen sind harmlose Hautwucherungen, die oft bei älteren Menschen auftreten.',
            'risk_level': 'Niedrig',
            'malignant': False,
            'precancerous': False,
            'prevalence': 'Häufig (besonders ab dem 50. Lebensjahr)',
            'treatment': 'Meist keine Behandlung nötig, Entfernung aus kosmetischen Gründen möglich',
            'emoji': '🟢'
        },
        'df': {
            'name': 'Dermatofibroma',
            'description': 'Dermatofibrome sind gutartige Knötchen in der Haut, oft als Reaktion auf kleinere Verletzungen.',
            'risk_level': 'Niedrig',
            'malignant': False,
            'precancerous': False,
            'prevalence': 'Häufig',
            'treatment': 'Keine Behandlung erforderlich, chirurgische Entfernung bei Bedarf',
            'emoji': '🟢'
        },
        'mel': {
            'name': 'Melanoma',
            'description': 'Das Melanom ist die gefährlichste Form von Hautkrebs mit hohem Metastasierungsrisiko.',
            'risk_level': 'Hoch',
            'malignant': True,
            'precancerous': False,
            'prevalence': 'Weniger häufig aber sehr gefährlich (ca. 5% der Hautkrebsfälle)',
            'treatment': 'Chirurgische Entfernung, Immuntherapie, zielgerichtete Therapie',
            'emoji': '🔴'
        },
        'nv': {
            'name': 'Melanocytic Nevi',
            'description': 'Melanozytäre Nävi sind gutartige Muttermale, die aus Melanozyten gebildet werden.',
            'risk_level': 'Sehr Niedrig',
            'malignant': False,
            'precancerous': False,
            'prevalence': 'Sehr häufig (fast jeder Mensch hat Muttermale)',
            'treatment': 'Regelmäßige Beobachtung, Entfernung bei Veränderungen',
            'emoji': '🟢'
        },
        'vasc': {
            'name': 'Vascular Lesions',
            'description': 'Vaskuläre Läsionen sind gutartige Veränderungen der Blutgefäße in der Haut.',
            'risk_level': 'Niedrig',
            'malignant': False,
            'precancerous': False,
            'prevalence': 'Häufig',
            'treatment': 'Laser-Therapie, Sklerotherapie bei kosmetischem Wunsch',
            'emoji': '🟢'
        }
    }

def apply_image_adjustments(image, brightness=1.0, contrast=1.0, rotation=0):
    """Apply adjustments to image"""
    # Brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)

    # Contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

    # Rotation
    if rotation != 0:
        image = image.rotate(rotation, expand=True)

    return image

def create_confidence_chart(results):
    """Create an interactive confidence chart"""
    df = pd.DataFrame(results)
    df['probability_percent'] = df['probability'] * 100

    # Create horizontal bar chart
    fig = px.bar(
        df,
        x='probability_percent',
        y='class_name',
        orientation='h',
        color='probability_percent',
        color_continuous_scale='RdYlGn',
        labels={'probability_percent': 'Confidence (%)', 'class_name': 'Condition'},
        title='Prediction Confidence Distribution'
    )

    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig

def create_risk_gauge(risk_level):
    """Create a gauge chart for risk level"""
    risk_values = {
        'Sehr Niedrig': 10,
        'Niedrig': 25,
        'Mittel': 50,
        'Mittel-Hoch': 75,
        'Hoch': 90
    }

    value = risk_values.get(risk_level, 50)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Risk Level"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig

def preprocess_image(image):
    """Preprocess image for model prediction using MobileNetV2 preprocessing"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array
    img_array = np.array(image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Apply MobileNetV2 preprocessing (scales to [-1, 1])
    img_array = preprocess_input(img_array)

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
st.title("🔬 Advanced Skin Cancer Classifier")
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

# Main content with tabs
st.markdown("---")

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Image Analysis", "🖼️ Batch Analysis", "📈 Prediction History", "📚 Condition Database"])

# TAB 1: Single Image Analysis
with tab1:
    st.header("📸 Single Image Analysis")

    col_upload, col_settings = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Wähle ein Bild einer Hautläsion aus",
            type=['jpg', 'jpeg', 'png'],
            help="Unterstützte Formate: JPG, JPEG, PNG",
            key="single_upload"
        )

    with col_settings:
        st.subheader("🎨 Bildbearbeitung")
        brightness = st.slider("Helligkeit", 0.5, 2.0, 1.0, 0.1, key="brightness_single")
        contrast = st.slider("Kontrast", 0.5, 2.0, 1.0, 0.1, key="contrast_single")
        rotation = st.slider("Rotation", -180, 180, 0, 15, key="rotation_single")

    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file)
        original_image = image.copy()

        # Apply adjustments
        adjusted_image = apply_image_adjustments(image, brightness, contrast, rotation)

        # Display images
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original / Angepasstes Bild")
            tab_orig, tab_adj = st.tabs(["Original", "Angepasst"])

            with tab_orig:
                st.image(original_image, use_container_width=True)
                st.caption(f"Bildgröße: {original_image.size[0]} x {original_image.size[1]} Pixel")

            with tab_adj:
                st.image(adjusted_image, use_container_width=True)
                st.caption(f"Helligkeit: {brightness}, Kontrast: {contrast}, Rotation: {rotation}°")

        with col2:
            st.subheader("Klassifizierung")

            # Predict button
            if st.button("🔍 Klassifizieren", type="primary", use_container_width=True, key="predict_single"):
                with st.spinner("Analysiere Bild..."):
                    # Make prediction using adjusted image
                    results = predict(model, adjusted_image, class_names)
                    condition_details = get_condition_details()

                    # Display top prediction
                    top_result = results[0]
                    confidence = top_result['probability'] * 100
                    detail = condition_details[top_result['class_key']]

                    # Color based on confidence
                    if confidence > 70:
                        color = "green"
                    elif confidence > 50:
                        color = "orange"
                    else:
                        color = "red"

                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;">
                        <h3 style="color: {color}; margin: 0;">{detail['emoji']} {detail['name']}</h3>
                        <p style="font-size: 32px; font-weight: bold; color: #000; margin: 10px 0;">{confidence:.1f}% <span style="font-size: 20px;">Konfidenz</span></p>
                        <p style="color: #555; margin: 0;">Klasse: {top_result['class_key']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add to history
                    history_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'prediction': detail['name'],
                        'class_key': top_result['class_key'],
                        'confidence': confidence,
                        'risk_level': detail['risk_level'],
                        'malignant': detail['malignant']
                    }
                    st.session_state.prediction_history.append(history_entry)

                    # Display detailed information
                    with st.expander("📋 Detaillierte Informationen"):
                        col_info1, col_info2 = st.columns(2)

                        with col_info1:
                            st.markdown(f"**Beschreibung:**\n{detail['description']}")
                            st.markdown(f"**Häufigkeit:** {detail['prevalence']}")

                        with col_info2:
                            st.markdown(f"**Risiko-Level:** {detail['risk_level']}")
                            st.markdown(f"**Bösartig:** {'Ja' if detail['malignant'] else 'Nein'}")
                            st.markdown(f"**Präkanzerös:** {'Ja' if detail['precancerous'] else 'Nein'}")
                            st.markdown(f"**Behandlung:** {detail['treatment']}")

                    # Interactive confidence chart
                    st.markdown("### 📊 Confidence Distribution")
                    fig = create_confidence_chart(results)
                    st.plotly_chart(fig, use_container_width=True, key="single_confidence_chart")

                    # Risk gauge
                    st.markdown("### ⚠️ Risk Assessment")
                    fig_gauge = create_risk_gauge(detail['risk_level'])
                    st.plotly_chart(fig_gauge, use_container_width=True, key="single_risk_gauge")

                    # Warning message
                    st.warning("""
                    ⚠️ **WICHTIGER HINWEIS**:
                    Diese Vorhersage dient nur zu Informationszwecken und ersetzt keine professionelle medizinische Diagnose.
                    Bei Verdacht auf Hautkrebs oder anderen Hautveränderungen konsultieren Sie bitte einen Dermatologen!
                    """)

    else:
        st.info("👆 Bitte laden Sie ein Bild hoch, um mit der Klassifizierung zu beginnen.")

# TAB 2: Batch Analysis
with tab2:
    st.header("🖼️ Batch Image Analysis")
    st.markdown("Laden Sie mehrere Bilder hoch, um sie gleichzeitig zu analysieren und zu vergleichen.")

    uploaded_files = st.file_uploader(
        "Wähle mehrere Bilder aus",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Unterstützte Formate: JPG, JPEG, PNG",
        key="batch_upload"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} Bilder hochgeladen")

        if st.button("🔍 Alle analysieren", type="primary", key="batch_predict"):
            batch_results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Analysiere Bild {idx + 1} von {len(uploaded_files)}...")
                progress_bar.progress((idx + 1) / len(uploaded_files))

                image = Image.open(file)
                results = predict(model, image, class_names)
                condition_details = get_condition_details()

                top_result = results[0]
                detail = condition_details[top_result['class_key']]

                batch_results.append({
                    'file_name': file.name,
                    'image': image,
                    'prediction': detail['name'],
                    'class_key': top_result['class_key'],
                    'confidence': top_result['probability'] * 100,
                    'risk_level': detail['risk_level'],
                    'emoji': detail['emoji'],
                    'malignant': detail['malignant']
                })

            status_text.text("✅ Analyse abgeschlossen!")
            progress_bar.empty()

            # Display results in a grid
            st.markdown("### Ergebnisse")

            cols_per_row = 3
            for i in range(0, len(batch_results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(batch_results):
                        result = batch_results[i + j]
                        with col:
                            st.image(result['image'], use_container_width=True)
                            st.markdown(f"**{result['emoji']} {result['prediction']}**")
                            st.markdown(f"Confidence: **{result['confidence']:.1f}%**")
                            st.markdown(f"Risk: {result['risk_level']}")

                            # Add to history
                            history_entry = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'prediction': result['prediction'],
                                'class_key': result['class_key'],
                                'confidence': result['confidence'],
                                'risk_level': result['risk_level'],
                                'malignant': result['malignant']
                            }
                            st.session_state.prediction_history.append(history_entry)

            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Batch Summary")

            summary_cols = st.columns(4)

            with summary_cols[0]:
                st.metric("Total Images", len(batch_results))

            with summary_cols[1]:
                avg_conf = sum(r['confidence'] for r in batch_results) / len(batch_results)
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")

            with summary_cols[2]:
                malignant_count = sum(1 for r in batch_results if r['malignant'])
                st.metric("Malignant Cases", malignant_count)

            with summary_cols[3]:
                unique_conditions = len(set(r['prediction'] for r in batch_results))
                st.metric("Unique Conditions", unique_conditions)

    else:
        st.info("👆 Bitte laden Sie mehrere Bilder hoch für die Batch-Analyse.")

# TAB 3: Prediction History
with tab3:
    st.header("📈 Prediction History")

    if st.session_state.prediction_history:
        df_history = pd.DataFrame(st.session_state.prediction_history)

        # Display statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", len(df_history))

        with col2:
            avg_conf = df_history['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")

        with col3:
            malignant_pct = (df_history['malignant'].sum() / len(df_history)) * 100
            st.metric("Malignant %", f"{malignant_pct:.1f}%")

        with col4:
            unique_predictions = df_history['prediction'].nunique()
            st.metric("Unique Conditions", unique_predictions)

        # Display history table
        st.markdown("### 📋 History Table")
        st.dataframe(
            df_history.style.background_gradient(subset=['confidence'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )

        # Visualizations
        st.markdown("### 📊 Visualizations")

        vis_col1, vis_col2 = st.columns(2)

        with vis_col1:
            # Prediction distribution
            fig_dist = px.pie(
                df_history,
                names='prediction',
                title='Prediction Distribution',
                hole=0.4
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="history_pie_chart")

        with vis_col2:
            # Confidence over time
            fig_time = px.line(
                df_history.reset_index(),
                x='index',
                y='confidence',
                title='Confidence Over Time',
                markers=True,
                labels={'index': 'Prediction Number', 'confidence': 'Confidence (%)'}
            )
            st.plotly_chart(fig_time, use_container_width=True, key="history_line_chart")

        # Export options
        st.markdown("### 💾 Export Options")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            csv = df_history.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with export_col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

    else:
        st.info("📭 Keine Vorhersagen in der Historie. Analysieren Sie Bilder, um Historie zu erstellen.")

# TAB 4: Condition Database
with tab4:
    st.header("📚 Condition Database")
    st.markdown("Umfassende Informationen über alle erkennbaren Hautläsionen.")

    condition_details = get_condition_details()

    for class_key, detail in condition_details.items():
        with st.expander(f"{detail['emoji']} {detail['name']} ({class_key})"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Beschreibung:**\n{detail['description']}")
                st.markdown(f"**Behandlung:** {detail['treatment']}")

            with col2:
                st.markdown(f"**Risiko-Level:** {detail['risk_level']}")
                st.markdown(f"**Häufigkeit:** {detail['prevalence']}")
                st.markdown(f"**Bösartig:** {'Ja' if detail['malignant'] else 'Nein'}")
                st.markdown(f"**Präkanzerös:** {'Ja' if detail['precancerous'] else 'Nein'}")

            # Risk gauge for each condition
            fig_gauge = create_risk_gauge(detail['risk_level'])
            st.plotly_chart(fig_gauge, use_container_width=True, key=f"condition_gauge_{class_key}")

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ Über diese App")

    st.markdown("""
    ### 🚀 Erweiterte Features
    - **Single Image Analysis**: Einzelbildanalyse mit Bildbearbeitung
    - **Batch Analysis**: Mehrere Bilder gleichzeitig analysieren
    - **Prediction History**: Alle Vorhersagen speichern und exportieren
    - **Condition Database**: Umfassende Informationen zu allen Erkrankungen

    ### Modell-Informationen
    - **Architektur**: MobileNetV2
    - **Methode**: Transfer Learning
    - **Datensatz**: HAM10000
    - **Anzahl Klassen**: 7
    - **Eingabegröße**: 224x224 Pixel

    ### 🎨 Bildbearbeitung
    Passen Sie Helligkeit, Kontrast und Rotation an, um optimale
    Ergebnisse zu erzielen.

    ### 📊 Visualisierungen
    - Interactive confidence charts
    - Risk assessment gauges
    - Prediction distribution analysis
    - Confidence trends over time

    ### Technologie
    - TensorFlow/Keras
    - MobileNetV2 (ImageNet Pre-Training)
    - Streamlit
    - Plotly (Interactive Charts)
    - Pandas (Data Analysis)

    ### HAM10000 Datensatz
    Der HAM10000 (Human Against Machine with 10000 training images)
    ist ein großer Datensatz dermatoskopischer Bilder verschiedener
    Hautläsionen, der für das Training von Machine Learning Modellen
    entwickelt wurde.
    """)

    st.markdown("---")

    # Statistics from history
    if st.session_state.prediction_history:
        st.markdown("### 📊 Session Stats")
        st.metric("Total Predictions", len(st.session_state.prediction_history))

    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit & TensorFlow")
    st.markdown("BADSA 2025 Group 3")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>Disclaimer</b>: Diese Anwendung dient ausschließlich zu Bildungs- und Demonstrationszwecken.
    Sie ist nicht für medizinische Diagnosen vorgesehen und sollte nicht als solche verwendet werden.
    Konsultieren Sie immer einen qualifizierten Arzt bei gesundheitlichen Bedenken.</p>
</div>
""", unsafe_allow_html=True)
