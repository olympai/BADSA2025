"""
Training script for skin cancer classification using MobileNetV2 with Transfer Learning
Dataset: HAM10000
"""

import os
import pandas as pd
import numpy as np
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Class labels
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
print("Skin Cancer Classification - MobileNetV2 Transfer Learning")
print("=" * 60)

# Load metadata
print("\n[1/8] Loading metadata...")
metadata = pd.read_csv('data/HAM10000_metadata.csv')
print(f"Total samples: {len(metadata)}")
print(f"\nClass distribution:")
print(metadata['dx'].value_counts())

# Create image paths
def get_image_path(image_id):
    """Find image path in either part_1 or part_2 directory"""
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
print("\n[2/8] Splitting data into train/validation/test sets...")
train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42, stratify=metadata['dx'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Calculate class weights for imbalanced dataset
class_weights_values = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_df['dx']),
    y=train_df['dx']
)
class_weights_dict = dict(enumerate(class_weights_values))
print("\nClass weights calculated for imbalanced dataset")

# Data augmentation for training
print("\n[3/8] Setting up data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Only rescaling for validation and test
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
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

# Build model with MobileNetV2
print("\n[4/8] Building MobileNetV2 model with Transfer Learning...")

# Load pre-trained MobileNetV2 (without top layer)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers for transfer learning
base_model.trainable = False
print(f"Base model loaded: {len(base_model.layers)} layers frozen")

# Build complete model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nModel architecture:")
model.summary()

# Callbacks
print("\n[5/8] Setting up training callbacks...")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Create models directory
os.makedirs('models', exist_ok=True)

# Train model
print("\n[6/8] Training model...")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

# Fine-tuning: Unfreeze last layers
print("\n[7/8] Fine-tuning: Unfreezing last layers...")
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

print(f"Unfrozen last 20 layers for fine-tuning")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(), keras.metrics.Recall()]
)

# Continue training with fine-tuning
print("Fine-tuning model...")
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

# Evaluate on test set
print("\n[8/8] Evaluating on test set...")
test_results = model.evaluate(test_generator, verbose=1)
print("\nTest Results:")
print(f"Loss: {test_results[0]:.4f}")
print(f"Accuracy: {test_results[1]:.4f}")
print(f"AUC: {test_results[2]:.4f}")
print(f"Precision: {test_results[3]:.4f}")
print(f"Recall: {test_results[4]:.4f}")

# Save final model
model.save('models/final_model.h5')
print("\nModel saved to 'models/final_model.h5'")

# Save class names
import json
with open('models/class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f, indent=2)
print("Class names saved to 'models/class_names.json'")

# Plot training history
print("\nGenerating training plots...")
plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# AUC
plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
print("Training plots saved to 'models/training_history.png'")

print("\n" + "=" * 60)
print("Training completed successfully!")
print("=" * 60)
