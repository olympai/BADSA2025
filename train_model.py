"""
Training script for skin cancer classification using MobileNetV2 with Transfer Learning
Dataset: HAM10000
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')
import logging
from datetime import datetime

# Setup logging
def setup_logging():
    """Configure logging to file and console"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("Logging initialized")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)

    return logger

# Initialize logger
logger = setup_logging()

# Configure TensorFlow for better memory management and to prevent mutex lock issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Reduce gRPC logging
os.environ['GRPC_TRACE'] = ''  # Disable gRPC tracing

# Force CPU-only for stability (Metal GPU can cause issues on Mac)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

# Additional settings to prevent mutex lock issues on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate OpenMP libraries
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Limit inter-op threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Limit intra-op threads

logger.info("Configuring TensorFlow...")

# Limit CPU threads to avoid lock issues (more aggressive settings)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Set memory growth to prevent allocation issues
try:
    physical_devices = tf.config.list_physical_devices('CPU')
    if physical_devices:
        logger.info(f"Found {len(physical_devices)} CPU device(s)")
except Exception as e:
    logger.warning(f"Could not configure physical devices: {e}")

logger.info("TensorFlow configured for CPU with single-threaded execution to prevent mutex locks")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration - Enhanced with adaptive learning and flexible batch sizes
IMG_SIZE = 224
BATCH_SIZE_TRAIN = 16  # Training batch size
BATCH_SIZE_VAL = 32    # Validation batch size (can be larger, no gradients)
BATCH_SIZE_TEST = 32   # Test batch size (can be larger, no gradients)
EPOCHS = 15  # Reduced from 20 for faster training
FINE_TUNE_EPOCHS = 5  # Reduced from 10
LEARNING_RATE = 0.0001

# Learning rate schedule options: 'reduce_on_plateau', 'cosine_decay', 'exponential_decay'
LR_SCHEDULE = 'cosine_decay'

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

logger.info("=" * 60)
logger.info("Skin Cancer Classification - MobileNetV2 Transfer Learning")
logger.info("=" * 60)

# Load metadata
logger.info("\n[1/8] Loading metadata...")
try:
    metadata = pd.read_csv('data/HAM10000_metadata.csv')
    logger.info(f"Total samples: {len(metadata)}")
    logger.info(f"\nClass distribution:")
    logger.info(f"\n{metadata['dx'].value_counts()}")
except Exception as e:
    logger.error(f"Failed to load metadata: {e}")
    raise

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
logger.info(f"Valid images found: {len(metadata)}")

# Split data
logger.info("\n[2/8] Splitting data into train/validation/test sets...")
train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42, stratify=metadata['dx'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

logger.info(f"Training samples: {len(train_df)}")
logger.info(f"Validation samples: {len(val_df)}")
logger.info(f"Test samples: {len(test_df)}")

# Calculate class weights for imbalanced dataset
class_weights_values = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_df['dx']),
    y=train_df['dx']
)
class_weights_dict = dict(enumerate(class_weights_values))
logger.info("\nClass weights calculated for imbalanced dataset:")
for idx, (class_name, weight) in enumerate(zip(sorted(np.unique(train_df['dx'])), class_weights_values)):
    logger.info(f"  {class_name}: {weight:.4f}")

# Data augmentation for training
logger.info("\n[3/8] Setting up data augmentation...")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Only rescaling for validation and test
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create data generators with different batch sizes
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE_TRAIN,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE_VAL,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE_TEST,
    class_mode='categorical',
    shuffle=False
)

logger.info(f"Batch sizes - Train: {BATCH_SIZE_TRAIN}, Val: {BATCH_SIZE_VAL}, Test: {BATCH_SIZE_TEST}")

logger.info(f"Classes: {train_generator.class_indices}")

# Build model with MobileNetV2
logger.info("\n[4/8] Building MobileNetV2 model with Transfer Learning...")

# Load pre-trained MobileNetV2 (without top layer)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers for transfer learning
base_model.trainable = False
logger.info(f"Base model loaded: {len(base_model.layers)} layers frozen")

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

# Setup learning rate schedule BEFORE compilation
logger.info("\nSetting up learning rate schedule...")
steps_per_epoch = len(train_generator)

if LR_SCHEDULE == 'cosine_decay':
    # Cosine decay with restarts
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=steps_per_epoch * 5,  # Restart every 5 epochs
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-7
    )
    logger.info(f"Using Cosine Decay with Restarts (restarts every 5 epochs)")
elif LR_SCHEDULE == 'exponential_decay':
    # Exponential decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=steps_per_epoch * 2,  # Decay every 2 epochs
        decay_rate=0.9,
        staircase=True
    )
    logger.info(f"Using Exponential Decay (decay every 2 epochs)")
else:
    # Standard constant learning rate (will use ReduceLROnPlateau callback)
    lr_schedule = LEARNING_RATE
    logger.info(f"Using constant LR with ReduceLROnPlateau callback")

# Compile model with adaptive learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(), keras.metrics.Recall()]
)

logger.info("\nModel architecture:")
model.summary()
logger.info(f"Total trainable parameters: {model.count_params():,}")

# Custom callback for per-class metrics
class PerClassMetrics(keras.callbacks.Callback):
    def __init__(self, validation_generator, class_names):
        super().__init__()
        self.validation_generator = validation_generator
        self.class_names = class_names
        self.history = {
            'precision_per_class': [],
            'recall_per_class': [],
            'f1_per_class': []
        }

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        self.validation_generator.reset()
        y_pred = self.model.predict(self.validation_generator, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.validation_generator.classes

        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average=None, zero_division=0
        )

        # Store metrics
        self.history['precision_per_class'].append(precision.tolist())
        self.history['recall_per_class'].append(recall.tolist())
        self.history['f1_per_class'].append(f1.tolist())

        # Log summary
        logger.info(f"\n--- Per-Class Metrics (Epoch {epoch + 1}) ---")
        for i, class_name in enumerate(sorted(self.class_names.keys())):
            logger.info(f"{class_name}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}")

# Callbacks
logger.info("\n[5/8] Setting up training callbacks...")

# Setup callback-based learning rate adjustment if using reduce_on_plateau
lr_callbacks = []
if LR_SCHEDULE == 'reduce_on_plateau':
    lr_callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    )

# Per-class metrics tracking
per_class_metrics = PerClassMetrics(val_generator, CLASS_NAMES)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_recall',
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
    per_class_metrics,
    keras.callbacks.CSVLogger('models/training_log.csv', append=False)
] + lr_callbacks

# Create models directory
os.makedirs('models', exist_ok=True)

# Train model
logger.info("\n[6/8] Training model...")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Batch sizes - Train: {BATCH_SIZE_TRAIN}, Val: {BATCH_SIZE_VAL}")
logger.info(f"Initial learning rate: {LEARNING_RATE}")
logger.info(f"Learning rate schedule: {LR_SCHEDULE}")
logger.info("Training started...")

try:
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    logger.info("Initial training phase completed successfully")
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# Fine-tuning: Unfreeze last layers
logger.info("\n[7/8] Fine-tuning: Unfreezing last layers...")
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

logger.info(f"Unfrozen last 20 layers for fine-tuning")
trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
logger.info(f"Trainable parameters in fine-tuning: {trainable_params:,}")

# Setup learning rate schedule for fine-tuning (lower learning rate)
fine_tune_lr = LEARNING_RATE / 10
if LR_SCHEDULE == 'cosine_decay':
    # Cosine decay with restarts for fine-tuning
    lr_schedule_fine = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=fine_tune_lr,
        first_decay_steps=steps_per_epoch * 3,  # Restart every 3 epochs
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-8
    )
    logger.info(f"Fine-tuning with Cosine Decay (LR={fine_tune_lr})")
elif LR_SCHEDULE == 'exponential_decay':
    # Exponential decay for fine-tuning
    lr_schedule_fine = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=fine_tune_lr,
        decay_steps=steps_per_epoch * 2,
        decay_rate=0.9,
        staircase=True
    )
    logger.info(f"Fine-tuning with Exponential Decay (LR={fine_tune_lr})")
else:
    # Constant learning rate for fine-tuning
    lr_schedule_fine = fine_tune_lr
    logger.info(f"Fine-tuning with constant LR={fine_tune_lr}")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_fine),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(), keras.metrics.Recall()]
)

# Continue training with fine-tuning
logger.info("Fine-tuning model...")
try:
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    logger.info("Fine-tuning phase completed successfully")
except Exception as e:
    logger.error(f"Fine-tuning failed: {e}")
    raise

# Evaluate on test set with detailed metrics
logger.info("\n[8/8] Evaluating on test set...")
test_results = model.evaluate(test_generator, verbose=1)
logger.info("\nOverall Test Results:")
logger.info(f"Loss: {test_results[0]:.4f}")
logger.info(f"Accuracy: {test_results[1]:.4f}")
logger.info(f"AUC: {test_results[2]:.4f}")
logger.info(f"Precision: {test_results[3]:.4f}")
logger.info(f"Recall: {test_results[4]:.4f}")

# Generate predictions for detailed analysis
logger.info("\nGenerating predictions for detailed analysis...")
test_generator.reset()
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Get class names in correct order
class_indices = train_generator.class_indices
class_names_ordered = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

# Classification Report
logger.info("\n" + "="*60)
logger.info("DETAILED CLASSIFICATION REPORT")
logger.info("="*60)
class_report = classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names_ordered,
    digits=4
)
logger.info(f"\n{class_report}")

# Save classification report
with open('models/classification_report.txt', 'w') as f:
    f.write("Classification Report - Test Set\n")
    f.write("="*60 + "\n")
    f.write(class_report)
logger.info("Classification report saved to 'models/classification_report.txt'")

# Per-class metrics dictionary
class_report_dict = classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names_ordered,
    output_dict=True
)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
logger.info("\n" + "="*60)
logger.info("CONFUSION MATRIX")
logger.info("="*60)
logger.info("Rows: True labels, Columns: Predicted labels")
logger.info(f"\n{pd.DataFrame(cm, index=class_names_ordered, columns=class_names_ordered)}")

# Save confusion matrix
np.save('models/confusion_matrix.npy', cm)
pd.DataFrame(cm, index=class_names_ordered, columns=class_names_ordered).to_csv(
    'models/confusion_matrix.csv'
)
logger.info("Confusion matrix saved to 'models/confusion_matrix.csv'")

# Calculate per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred_classes, average=None, zero_division=0
)

# Create detailed metrics dataframe
per_class_results = pd.DataFrame({
    'class': class_names_ordered,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'support': support
})

logger.info("\n" + "="*60)
logger.info("PER-CLASS METRICS SUMMARY")
logger.info("="*60)
logger.info(f"\n{per_class_results.to_string(index=False)}")

# Save per-class metrics
per_class_results.to_csv('models/per_class_metrics.csv', index=False)
logger.info("\nPer-class metrics saved to 'models/per_class_metrics.csv'")

# Save final model
model.save('models/final_model.h5')
logger.info("\nModel saved to 'models/final_model.h5'")

# Save comprehensive metrics in JSON format
import json

# Combine all metrics
all_metrics = {
    'test_metrics': {
        'loss': float(test_results[0]),
        'accuracy': float(test_results[1]),
        'auc': float(test_results[2]),
        'precision': float(test_results[3]),
        'recall': float(test_results[4])
    },
    'per_class_metrics': {
        class_name: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
        for i, class_name in enumerate(class_names_ordered)
    },
    'confusion_matrix': cm.tolist(),
    'class_names': class_names_ordered,
    'class_descriptions': CLASS_NAMES,
    'configuration': {
        'img_size': IMG_SIZE,
        'batch_size_train': BATCH_SIZE_TRAIN,
        'batch_size_val': BATCH_SIZE_VAL,
        'batch_size_test': BATCH_SIZE_TEST,
        'epochs': EPOCHS,
        'fine_tune_epochs': FINE_TUNE_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'lr_schedule': LR_SCHEDULE
    },
    'training_history': {
        'per_class_precision': per_class_metrics.history['precision_per_class'],
        'per_class_recall': per_class_metrics.history['recall_per_class'],
        'per_class_f1': per_class_metrics.history['f1_per_class']
    }
}

with open('models/all_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
logger.info("\nAll metrics saved to 'models/all_metrics.json'")

with open('models/class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f, indent=2)
logger.info("Class names saved to 'models/class_names.json'")

# Plot training history
logger.info("\nGenerating training plots...")

# Overall training metrics
fig = plt.figure(figsize=(20, 10))

# Accuracy
plt.subplot(2, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
plt.title('Model Accuracy', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(2, 3, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
plt.title('Model Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# AUC
plt.subplot(2, 3, 3)
plt.plot(history.history['auc'], label='Train AUC', marker='o')
plt.plot(history.history['val_auc'], label='Val AUC', marker='s')
plt.title('Model AUC', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True, alpha=0.3)

# Precision
plt.subplot(2, 3, 4)
plt.plot(history.history['precision'], label='Train Precision', marker='o')
plt.plot(history.history['val_precision'], label='Val Precision', marker='s')
plt.title('Model Precision', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.grid(True, alpha=0.3)

# Recall
plt.subplot(2, 3, 5)
plt.plot(history.history['recall'], label='Train Recall', marker='o')
plt.plot(history.history['val_recall'], label='Val Recall', marker='s')
plt.title('Model Recall', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(True, alpha=0.3)

# Per-class F1 over epochs (final epoch)
plt.subplot(2, 3, 6)
final_f1 = per_class_metrics.history['f1_per_class'][-1]
bars = plt.bar(range(len(class_names_ordered)), final_f1, alpha=0.7)
plt.title('Final F1-Score per Class', fontsize=12, fontweight='bold')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.xticks(range(len(class_names_ordered)), class_names_ordered, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Color bars by performance
for i, (bar, f1_val) in enumerate(zip(bars, final_f1)):
    if f1_val >= 0.8:
        bar.set_color('green')
    elif f1_val >= 0.6:
        bar.set_color('orange')
    else:
        bar.set_color('red')

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
logger.info("Training plots saved to 'models/training_history.png'")

# Plot confusion matrix
plt.figure(figsize=(12, 10))
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=20)
plt.colorbar(im, fraction=0.046, pad=0.04)

tick_marks = np.arange(len(class_names_ordered))
plt.xticks(tick_marks, class_names_ordered, rotation=45, ha='right')
plt.yticks(tick_marks, class_names_ordered)

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10)

plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
logger.info("Confusion matrix plot saved to 'models/confusion_matrix.png'")

# Plot per-class metrics comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Precision comparison
axes[0].barh(class_names_ordered, precision, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Precision', fontsize=11)
axes[0].set_title('Precision per Class', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].set_xlim([0, 1])

# Recall comparison
axes[1].barh(class_names_ordered, recall, alpha=0.7, color='darkorange')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_title('Recall per Class', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].set_xlim([0, 1])

# F1-Score comparison
axes[2].barh(class_names_ordered, f1, alpha=0.7, color='forestgreen')
axes[2].set_xlabel('F1-Score', fontsize=11)
axes[2].set_title('F1-Score per Class', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')
axes[2].set_xlim([0, 1])

plt.tight_layout()
plt.savefig('models/per_class_metrics.png', dpi=150, bbox_inches='tight')
logger.info("Per-class metrics plot saved to 'models/per_class_metrics.png'")

# Plot per-class metrics evolution over epochs
if len(per_class_metrics.history['f1_per_class']) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    epochs_range = range(1, len(per_class_metrics.history['f1_per_class']) + 1)

    # Precision evolution
    for i, class_name in enumerate(class_names_ordered):
        precision_vals = [epoch_data[i] for epoch_data in per_class_metrics.history['precision_per_class']]
        axes[0].plot(epochs_range, precision_vals, marker='o', label=class_name, alpha=0.7)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Precision', fontsize=11)
    axes[0].set_title('Precision Evolution per Class', fontsize=12, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Recall evolution
    for i, class_name in enumerate(class_names_ordered):
        recall_vals = [epoch_data[i] for epoch_data in per_class_metrics.history['recall_per_class']]
        axes[1].plot(epochs_range, recall_vals, marker='s', label=class_name, alpha=0.7)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Recall', fontsize=11)
    axes[1].set_title('Recall Evolution per Class', fontsize=12, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # F1 evolution
    for i, class_name in enumerate(class_names_ordered):
        f1_vals = [epoch_data[i] for epoch_data in per_class_metrics.history['f1_per_class']]
        axes[2].plot(epochs_range, f1_vals, marker='^', label=class_name, alpha=0.7)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('F1-Score', fontsize=11)
    axes[2].set_title('F1-Score Evolution per Class', fontsize=12, fontweight='bold')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/per_class_metrics_evolution.png', dpi=150, bbox_inches='tight')
    logger.info("Per-class metrics evolution plot saved to 'models/per_class_metrics_evolution.png'")

logger.info("\n" + "=" * 60)
logger.info("Training completed successfully!")
logger.info(f"Total training time: Check log file for timestamps")
logger.info("=" * 60)