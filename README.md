# Hautkrebs-Klassifizierung mit MobileNetV2

Dieses Projekt verwendet Transfer Learning mit MobileNetV2, um Hautläsionen zu klassifizieren. Das Modell wurde auf dem HAM10000-Datensatz trainiert und kann 7 verschiedene Arten von Hautläsionen identifizieren.

## Übersicht

- **Modell**: MobileNetV2 (vortrainiert auf ImageNet), sowie ResNet in separatem Notebook
- **Methode**: Transfer Learning mit Fine-Tuning
- **Datensatz**: HAM10000 (ca. 10.000 dermatoskopische Bilder)
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

### Callbacks
- **ModelCheckpoint**: Speichert bestes Modell basierend auf Validation Accuracy
- **EarlyStopping**: Stoppt Training bei Stagnation (Patience: 5 Epochen)
- **ReduceLROnPlateau**: Reduziert Learning Rate bei Plateau (Patience: 3 Epochen)

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
