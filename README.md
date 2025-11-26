# Hautkrebs-Klassifizierung mit MobileNetV2

Dieses Projekt verwendet **Transfer Learning** mit **MobileNetV2**, um Hautläsionen zu klassifizieren. Das Modell wurde auf dem **HAM10000-Datensatz** trainiert und kann 7 verschiedene Arten von Hautläsionen identifizieren.

## Übersicht

- **Modell**: MobileNetV2 (vortrainiert auf ImageNet)
- **Methode**: Transfer Learning mit Fine-Tuning
- **Datensatz**: HAM10000 (~10.000 dermatoskopische Bilder)
- **Klassen**: 7 verschiedene Hautläsionstypen
- **Interface**: Streamlit Web-Anwendung

## Klassifizierte Hautläsionen

Das Modell kann folgende Hautläsionen klassifizieren:

1. **Actinic keratoses (akiec)** - Aktinische Keratosen
2. **Basal cell carcinoma (bcc)** - Basalzellkarzinom
3. **Benign keratosis-like lesions (bkl)** - Gutartige keratoseartige Läsionen
4. **Dermatofibroma (df)** - Dermatofibrom
5. **Melanoma (mel)** - Melanom
6. **Melanocytic nevi (nv)** - Melanozytäre Nävi (Muttermale)
7. **Vascular lesions (vasc)** - Vaskuläre Läsionen

## Installation

### 1. Repository klonen (falls noch nicht geschehen)

```bash
git clone <repository-url>
cd BADSA2025
```

### 2. Virtuelle Umgebung erstellen (empfohlen)

```bash
python -m venv venv
source venv/bin/activate  # Auf Windows: venv\Scripts\activate
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

## Datenstruktur

Der HAM10000-Datensatz sollte folgende Struktur haben:

```
BADSA2025/
├── data/
│   ├── HAM10000_images_part_1/
│   │   └── *.jpg
│   ├── HAM10000_images_part_2/
│   │   └── *.jpg
│   └── HAM10000_metadata.csv
├── train_model.py
├── app.py
└── requirements.txt
```

## Verwendung

### Schritt 1: Modell trainieren

Trainiere das MobileNetV2-Modell auf dem HAM10000-Datensatz:

```bash
python train_model.py
```

**Was passiert beim Training:**
- Lädt HAM10000-Metadaten und Bilder
- Split in Train (70%), Validation (15%), Test (15%)
- Data Augmentation für Training
- Transfer Learning mit vortrainiertem MobileNetV2
- Training für 20 Epochen
- Fine-Tuning der letzten 20 Layer für weitere 10 Epochen
- Evaluation auf Test-Set
- Speichert Modell in `models/final_model.h5`

**Training dauert je nach Hardware:**
- Mit GPU: ca. 30-60 Minuten
- Nur CPU: 2-4 Stunden

**Ausgaben nach dem Training:**
- `models/final_model.h5` - Das trainierte Modell
- `models/best_model.h5` - Das beste Modell während des Trainings
- `models/class_names.json` - Klassennamen
- `models/training_history.png` - Trainingsplots (Accuracy, Loss, AUC)

### Schritt 2: Streamlit-App starten

Starte die Web-Anwendung:

```bash
streamlit run app.py
```

Die App öffnet sich automatisch im Browser (normalerweise unter `http://localhost:8501`).

### Schritt 3: Bilder klassifizieren

1. Lade ein Bild einer Hautläsion hoch (JPG, JPEG oder PNG)
2. Klicke auf "Klassifizieren"
3. Erhalte eine Vorhersage mit Wahrscheinlichkeiten für alle Klassen

## Features des Trainingsskripts

### Transfer Learning
- Nutzt vortrainierte MobileNetV2-Weights (ImageNet)
- Friert anfangs alle Base-Layer ein
- Trainiert nur die neuen Top-Layer
- Fine-Tuning: Entfriert die letzten 20 Layer für bessere Performance

### Data Augmentation
- Rotation (20°)
- Width/Height Shift (20%)
- Horizontal & Vertical Flip
- Zoom (20%)

### Umgang mit Klassenungleichgewicht
- Berechnet Class Weights automatisch
- Balanciert Training für unterrepräsentierte Klassen

### Callbacks
- **ModelCheckpoint**: Speichert bestes Modell basierend auf Validation Accuracy
- **EarlyStopping**: Stoppt Training bei Stagnation (Patience: 5 Epochen)
- **ReduceLROnPlateau**: Reduziert Learning Rate bei Plateau (Patience: 3 Epochen)

### Metriken
- Accuracy
- AUC (Area Under Curve)
- Precision
- Recall

## Features der Streamlit-App

- Intuitive Benutzeroberfläche
- Drag & Drop oder Click zum Upload
- Bildvorschau
- Top-Vorhersage mit farbkodierter Konfidenz
- Alle Klassifikationswahrscheinlichkeiten
- Detaillierte Informationen zu Klassen
- Medizinischer Disclaimer

## Modell-Architektur

```
MobileNetV2 (vortrainiert)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.5)
    ↓
Dense (256, relu)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense (7, softmax)
```

## Hyperparameter

| Parameter | Wert |
|-----------|------|
| Bildgröße | 224x224 |
| Batch Size | 32 |
| Initial Epochs | 20 |
| Fine-Tuning Epochs | 10 |
| Learning Rate (Initial) | 0.0001 |
| Learning Rate (Fine-Tuning) | 0.00001 |
| Optimizer | Adam |

## Erwartete Performance

Basierend auf dem HAM10000-Datensatz können Sie erwarten:

- **Test Accuracy**: 75-85%
- **AUC**: 0.85-0.95

Die Performance variiert je nach:
- Trainingsdauer
- Hardware
- Random Seeds
- Klassenverteilung

## Wichtige Hinweise

### Medizinischer Disclaimer
Diese Anwendung dient **ausschließlich zu Bildungs- und Demonstrationszwecken**.
Sie ist **NICHT** für medizinische Diagnosen vorgesehen und sollte **NICHT** als solche verwendet werden.

Bei Verdacht auf Hautkrebs oder anderen Hautveränderungen konsultieren Sie **immer einen qualifizierten Dermatologen**!

### Limitierungen
- Das Modell wurde nur auf dermatoskopischen Bildern trainiert
- Performance kann bei anderen Bildtypen abweichen
- Keine Echtzeit-Validierung durch medizinisches Fachpersonal
- Klassenungleichgewicht im Datensatz kann Vorhersagen beeinflussen

## Troubleshooting

### Problem: "Model not found"
**Lösung**: Trainiere zuerst das Modell mit `python train_model.py`

### Problem: Speicherfehler beim Training
**Lösung**: Reduziere `BATCH_SIZE` in `train_model.py` (z.B. auf 16 oder 8)

### Problem: Training sehr langsam
**Lösung**:
- Prüfe ob TensorFlow GPU-Unterstützung hat: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Reduziere `EPOCHS` für schnellere Tests
- Nutze kleinere Bildgröße (z.B. 128x128 statt 224x224)

### Problem: Streamlit-App lädt nicht
**Lösung**:
- Prüfe ob Port 8501 frei ist
- Starte mit anderem Port: `streamlit run app.py --server.port 8502`

## Datensatz-Quelle

**HAM10000 (Human Against Machine with 10000 training images)**

- Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).
- [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

## Technologie-Stack

- **Python 3.8+**
- **TensorFlow/Keras** - Deep Learning Framework
- **MobileNetV2** - CNN-Architektur
- **Streamlit** - Web-Interface
- **NumPy & Pandas** - Datenverarbeitung
- **Pillow** - Bildverarbeitung
- **Scikit-learn** - ML-Utilities
- **Matplotlib** - Visualisierung

## Projektstruktur

```
BADSA2025/
├── data/                          # Datensatz
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   └── HAM10000_metadata.csv
├── models/                        # Gespeicherte Modelle (nach Training)
│   ├── final_model.h5
│   ├── best_model.h5
│   ├── class_names.json
│   └── training_history.png
├── train_model.py                 # Trainingsskript
├── app.py                         # Streamlit-Anwendung
├── requirements.txt               # Python-Dependencies
└── README.md                      # Diese Datei
```

## Weiterführende Verbesserungen

Mögliche Erweiterungen:

1. **Ensemble-Learning**: Kombiniere mehrere Modelle (EfficientNet, ResNet, etc.)
2. **Grad-CAM**: Visualisierung, welche Bildbereiche für die Klassifikation wichtig sind
3. **Augmented Reality**: Mobile App mit Kamera-Integration
4. **Erklärbarkeit**: LIME oder SHAP für bessere Interpretierbarkeit
5. **Deployment**: Docker-Container oder Cloud-Deployment (AWS, GCP, Azure)
6. **Mehr Daten**: Ergänzung mit zusätzlichen Datensätzen
7. **Multi-Task Learning**: Gleichzeitige Vorhersage mehrerer Attribute

## Lizenz

Dieses Projekt dient Bildungszwecken. Bitte beachten Sie die Lizenzbedingungen des HAM10000-Datensatzes bei kommerzieller Nutzung.

## Kontakt

Bei Fragen oder Problemen erstellen Sie bitte ein Issue im Repository.

---

**Viel Erfolg beim Training und Klassifizieren! 🔬**
