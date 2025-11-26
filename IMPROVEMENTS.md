# Verbesserungsmöglichkeiten für Hautkrebs-Klassifizierung

## ✅ Bereits implementiert (train_model_improved.py)
- [x] Dropout reduziert (0.5→0.3, 0.3→0.2)
- [x] Dense Layer vergrößert (256→512)
- [x] Data Augmentation reduziert
- [x] Mehr Epochen (15→25, 5→10)
- [x] Mehr Layers entfroren (20→40)

## 🚀 Weitere Verbesserungen

### 1. **Architektur-Alternativen**
```python
# EfficientNetB0 (besser als MobileNetV2)
from tensorflow.keras.applications import EfficientNetB0
base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
# Erwartung: +2-5% Accuracy
```

```python
# ResNet50 (größer aber genauer)
from tensorflow.keras.applications import ResNet50V2
base_model = ResNet50V2(...)
# Erwartung: +3-6% Accuracy
```

### 2. **Learning Rate Scheduling**
```python
# Cosine Annealing
keras.callbacks.CosineDecayRestarts(
    initial_learning_rate=0.0001,
    first_decay_steps=1000,
    t_mul=2.0,
    m_mul=0.9,
    alpha=0.0
)
# Erwartung: +1-3% Accuracy
```

### 3. **Progressive Resizing**
```python
# Start: 128x128 für 10 Epochen
# Dann: 224x224 für Rest
# Schneller Training + bessere Konvergenz
# Erwartung: -30% Training Zeit, +1-2% Accuracy
```

### 4. **Mixup / CutMix Augmentation**
```python
# Moderne Augmentation-Techniken
# Mische zwei Bilder + Labels
def mixup(image1, image2, label1, label2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    mixed_image = lam * image1 + (1 - lam) * image2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_image, mixed_label
# Erwartung: +2-4% Accuracy
```

### 5. **Focal Loss für Klassenungleichgewicht**
```python
# Besser als Class Weights
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

model.compile(
    optimizer=...,
    loss=FocalLoss(),
    metrics=[...]
)
# Erwartung: +2-3% für seltene Klassen
```

### 6. **Ensemble Learning**
```python
# Trainiere 3-5 verschiedene Modelle
models = [
    load_model('mobilenetv2.h5'),
    load_model('efficientnet.h5'),
    load_model('resnet.h5')
]

# Durchschnitts-Vorhersage
def ensemble_predict(image):
    predictions = [model.predict(image) for model in models]
    return np.mean(predictions, axis=0)
# Erwartung: +3-5% Accuracy
```

### 7. **Test-Time Augmentation (TTA)**
```python
# Bei Inferenz: Vorhersage mit verschiedenen Augmentationen
def predict_with_tta(model, image, n_augmentations=5):
    predictions = []
    for _ in range(n_augmentations):
        aug_image = augment(image)  # Flip, rotate, etc.
        pred = model.predict(aug_image)
        predictions.append(pred)
    return np.mean(predictions, axis=0)
# Erwartung: +1-2% Accuracy ohne Retraining
```

### 8. **Attention Mechanisms**
```python
# Squeeze-and-Excitation Blocks
class SEBlock(layers.Layer):
    def __init__(self, ratio=16):
        super().__init__()
        self.ratio = ratio

    def call(self, x):
        filters = x.shape[-1]
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(filters // self.ratio, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, filters))(se)
        return x * se

# Add nach base_model
model = keras.Sequential([
    base_model,
    SEBlock(),  # Attention!
    layers.GlobalAveragePooling2D(),
    ...
])
# Erwartung: +2-3% Accuracy
```

### 9. **Multi-Scale Training**
```python
# Trainiere mit verschiedenen Auflösungen
scales = [192, 224, 256]
for epoch in range(EPOCHS):
    scale = np.random.choice(scales)
    # Resize images to scale x scale
    train_on_scale(model, scale)
# Erwartung: +1-2% Robustheit
```

### 10. **Pseudo-Labeling (Semi-Supervised)**
```python
# 1. Trainiere auf gelabelten Daten
# 2. Vorhersage auf ungelabelten Daten
# 3. Nutze Konfidente Vorhersagen als neue Labels
# 4. Retrain

if confidence > 0.95:
    pseudo_labels.append((image, predicted_label))
# Erwartung: +3-5% mit zusätzlichen Daten
```

### 11. **Better Preprocessing**
```python
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
import cv2
def preprocess_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
# Erwartung: +1-2% für medizinische Bilder
```

### 12. **Knowledge Distillation**
```python
# Train großes Teacher-Modell (z.B. EfficientNetB4)
# Dann: Distill zu kleinem Student (MobileNetV2)
# Student lernt von Teacher's Soft Predictions

def distillation_loss(y_true, y_pred, teacher_pred, temperature=3):
    student_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    distill_loss = keras.losses.categorical_crossentropy(
        tf.nn.softmax(teacher_pred / temperature),
        tf.nn.softmax(y_pred / temperature)
    )
    return 0.5 * student_loss + 0.5 * distill_loss
# Erwartung: +2-3% bei kleinem Modell
```

## 📊 Erwartete Gesamt-Verbesserung

| Methode | Complexity | Expected Gain | Training Time |
|---------|-----------|---------------|---------------|
| Improved Script | Low | +5-10% | +50% |
| EfficientNet | Low | +2-5% | +30% |
| Focal Loss | Low | +2-3% | Same |
| TTA | Very Low | +1-2% | Same (inference) |
| Ensemble | Medium | +3-5% | 3-5x |
| Mixup/Cutmix | Medium | +2-4% | +20% |
| Attention | Medium | +2-3% | +10% |
| All Combined | High | +15-25% | 5x |

## 🎯 Quick Wins (einfach zu implementieren)

1. **train_model_improved.py nutzen** (bereits fertig!)
2. **Test-Time Augmentation** (keine Änderung am Training)
3. **EfficientNetB0** statt MobileNetV2
4. **Focal Loss** statt normale Cross-Entropy

## 💡 Best Practice Kombination

Für optimale Ergebnisse:
```
EfficientNetB0
+ Reduced Dropout (0.3, 0.2)
+ Focal Loss
+ Mixup Augmentation
+ TTA bei Inference
+ Ensemble von 3 Modellen

Expected: 75-80% Accuracy, 93-95% AUC
```

## 🔍 Debugging & Analysis Tools

```python
# 1. Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
cm = confusion_matrix(y_true, y_pred_classes)
print(classification_report(y_true, y_pred_classes))

# 2. Grad-CAM (welche Bereiche wichtig sind)
import tf_keras_vis
from tf_keras_vis.gradcam import Gradcam
gradcam = Gradcam(model)
cam = gradcam(loss, image, penultimate_layer=-1)

# 3. Learning Rate Finder
import keras_tuner
def find_lr(model, train_gen):
    lrs = []
    losses = []
    for lr in np.logspace(-6, -2, 100):
        model.optimizer.learning_rate = lr
        loss = model.fit(train_gen, epochs=1).history['loss'][0]
        lrs.append(lr)
        losses.append(loss)
    plt.plot(np.log10(lrs), losses)
    plt.show()
```

## 📚 Resources

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)
- [Mixup Paper](https://arxiv.org/abs/1710.09412)
- [Test-Time Augmentation](https://stepup.ai/test_time_data_augmentation/)
- [HAM10000 Leaderboard](https://challenge.isic-archive.com/leaderboards/)

## ⚠️ Important Notes

- Immer mit Validation Set testen vor Test Set
- Cross-Validation für robuste Ergebnisse
- Nie Test Set für Hyperparameter-Tuning nutzen
- Dokumentiere alle Experimente (MLflow, WandB)
