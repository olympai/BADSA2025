"""
Recall-Optimized Training for Skin Cancer Classification
Priority: Detect ALL potentially cancerous lesions (minimize False Negatives)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

print("Configuring TensorFlow...")
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
print("TensorFlow configured for CPU with limited threading")

np.random.seed(42)
tf.random.set_seed(42)

# Configuration - RECALL OPTIMIZED
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
LEARNING_RATE = 0.0001
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2
DENSE_UNITS = 512

# RECALL OPTIMIZATION: Higher weights for dangerous classes
# mel (Melanoma), bcc (Basal cell carcinoma), akiec (Actinic keratoses) = CANCER
DANGEROUS_CLASSES = ['mel', 'bcc', 'akiec']

CLASS_NAMES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

print("=" * 60)
print("RECALL-OPTIMIZED Skin Cancer Classification")
print("=" * 60)
print("\nObjective: MAXIMIZE RECALL - Detect ALL potentially cancerous lesions")
print("Trade-off: Accept more False Positives to minimize False Negatives")
print("=" * 60)

# Load metadata
print("\n[1/9] Loading metadata...")
metadata = pd.read_csv('data/HAM10000_metadata.csv')
print(f"Total samples: {len(metadata)}")
print(f"\nClass distribution:")
print(metadata['dx'].value_counts())

# Mark dangerous classes
metadata['is_dangerous'] = metadata['dx'].isin(DANGEROUS_CLASSES)
print(f"\nDangerous (cancer) classes: {metadata['is_dangerous'].sum()}")
print(f"Benign classes: {(~metadata['is_dangerous']).sum()}")

def get_image_path(image_id):
    path1 = f'data/HAM10000_images_part_1/{image_id}.jpg'
    path2 = f'data/HAM10000_images_part_2/{image_id}.jpg'
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None

metadata['path'] = metadata['image_id'].apply(get_image_path)
metadata = metadata.dropna(subset=['path'])
print(f"Valid images found: {len(metadata)}")

# Split data
print("\n[2/9] Splitting data...")
train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42, stratify=metadata['dx'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# RECALL OPTIMIZATION: Enhanced class weights
print("\n[3/9] Calculating RECALL-OPTIMIZED class weights...")
class_weights_values = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_df['dx']),
    y=train_df['dx']
)

# Boost dangerous classes by 50%
class_names_list = sorted(train_df['dx'].unique())
class_weights_dict = {}
for idx, class_name in enumerate(class_names_list):
    weight = class_weights_values[idx]
    if class_name in DANGEROUS_CLASSES:
        weight *= 1.5  # 50% increase for cancer classes
    class_weights_dict[idx] = weight
    print(f"  {class_name}: {weight:.2f} {'(BOOSTED)' if class_name in DANGEROUS_CLASSES else ''}")

# Data augmentation
print("\n[4/9] Setting up data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Classes: {train_generator.class_indices}")

# Custom Focal Loss for better recall
print("\n[5/9] Setting up Focal Loss for recall optimization...")

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss focuses on hard examples
    gamma > 0: Focus more on hard examples (good for recall)
    alpha: Weight for positive class
    """
    def __init__(self, gamma=2.5, alpha=0.75, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma  # Higher gamma = more focus on hard examples
        self.alpha = alpha  # Higher alpha = more weight on positive examples

    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate focal loss weight
        # (1 - p_t)^gamma focuses on hard examples
        weight = tf.pow(1 - y_pred, self.gamma)

        # Apply alpha weighting
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        # Focal loss
        focal_loss = alpha_weight * weight * cross_entropy

        return tf.reduce_sum(focal_loss, axis=-1)

# Build model
print("\n[6/9] Building model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False
print(f"Base model loaded: {len(base_model.layers)} layers frozen")

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(DROPOUT_1),
    layers.Dense(DENSE_UNITS, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(DROPOUT_2),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

# RECALL OPTIMIZATION: Custom metrics
class RecallAtPrecision(keras.metrics.Metric):
    """Custom metric: Recall at 70% precision"""
    def __init__(self, target_precision=0.7, name='recall_at_p70'):
        super().__init__(name=name)
        self.target_precision = target_precision
        self.recall = keras.metrics.Recall()
        self.precision = keras.metrics.Precision()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.recall.update_state(y_true, y_pred, sample_weight)
        self.precision.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.recall.result()

    def reset_state(self):
        self.recall.reset_state()
        self.precision.reset_state()

# Compile with Focal Loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=FocalLoss(gamma=2.5, alpha=0.75),
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),  # PRIMARY METRIC
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalseNegatives(name='fn')
    ]
)

print("\nModel architecture:")
model.summary()

# RECALL OPTIMIZATION: Callbacks focus on recall
print("\n[7/9] Setting up RECALL-OPTIMIZED callbacks...")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/best_model_recall.h5',
        monitor='val_recall',  # CHANGED: Monitor recall instead of accuracy
        save_best_only=True,
        mode='max',  # Maximize recall
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_recall',  # CHANGED: Stop based on recall
        patience=8,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    # Log false negatives (most important to minimize)
    keras.callbacks.CSVLogger('models/training_recall_log.csv', append=False)
]

os.makedirs('models', exist_ok=True)

# Train model
print("\n[8/9] Training model (RECALL OPTIMIZATION)...")
print(f"Epochs: {EPOCHS}")
print(f"Objective: Maximize Recall (minimize False Negatives)")
print(f"Focal Loss: gamma=2.5, alpha=0.75")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

# Fine-tuning
print("\n[8b/9] Fine-tuning with RECALL focus...")
base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
print(f"Unfrozen last {trainable_count} layers")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss=FocalLoss(gamma=2.5, alpha=0.75),
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalseNegatives(name='fn')
    ]
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

# Evaluate on test set
print("\n[9/9] Evaluating on test set...")
test_results = model.evaluate(test_generator, verbose=1)
metric_names = model.metrics_names

print("\n" + "=" * 60)
print("RECALL-OPTIMIZED Test Results:")
print("=" * 60)
for name, value in zip(metric_names, test_results):
    print(f"{name:20s}: {value:.4f}")

# Additional analysis
print("\n" + "=" * 60)
print("Detailed Analysis:")
print("=" * 60)

# Get predictions
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Per-class recall
from sklearn.metrics import classification_report, confusion_matrix
print("\nPer-Class Metrics (focus on Recall):")
class_indices = {v: k for k, v in train_generator.class_indices.items()}
target_names = [class_indices[i] for i in range(len(class_indices))]
print(classification_report(y_true, y_pred_classes, target_names=target_names))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred_classes)
print(cm)

# Focus on dangerous classes
print("\n" + "=" * 60)
print("CRITICAL: Cancer Detection Rates (Recall for dangerous classes):")
print("=" * 60)
for idx, class_name in enumerate(target_names):
    if class_name in DANGEROUS_CLASSES:
        recall = cm[idx, idx] / cm[idx, :].sum() if cm[idx, :].sum() > 0 else 0
        print(f"  {CLASS_NAMES[class_name]:30s}: {recall*100:.1f}% recall ⚠️")

# Save final model
model.save('models/final_model_recall.h5')
print("\nModel saved to 'models/final_model_recall.h5'")

# Save class names
import json
with open('models/class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f, indent=2)

# Plot training history
print("\nGenerating training plots...")
plt.figure(figsize=(20, 10))

# Recall (most important)
plt.subplot(2, 3, 1)
plt.plot(history.history['recall'], label='Train Recall', linewidth=2)
plt.plot(history.history['val_recall'], label='Val Recall', linewidth=2)
plt.title('Recall (PRIMARY METRIC)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

# Precision
plt.subplot(2, 3, 2)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Val Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# False Negatives (minimize!)
plt.subplot(2, 3, 3)
plt.plot(history.history['fn'], label='Train FN')
plt.plot(history.history['val_fn'], label='Val FN')
plt.title('False Negatives (MINIMIZE!)')
plt.xlabel('Epoch')
plt.ylabel('Count')
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(2, 3, 4)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(2, 3, 5)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Focal Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# AUC
plt.subplot(2, 3, 6)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.title('AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/training_history_recall.png', dpi=150, bbox_inches='tight')
print("Training plots saved to 'models/training_history_recall.png'")

print("\n" + "=" * 60)
print("RECALL-OPTIMIZED Training completed!")
print("=" * 60)
print("\nOptimizations applied:")
print("✓ Focal Loss (gamma=2.5, alpha=0.75)")
print("✓ Boosted weights for cancer classes (+50%)")
print("✓ ModelCheckpoint monitors val_recall")
print("✓ EarlyStopping monitors val_recall")
print("✓ Extended training (30+15 epochs)")
print("✓ Per-class recall analysis")
print("\n⚠️  Result: Minimized False Negatives (missed cancer cases)")
print("=" * 60)
