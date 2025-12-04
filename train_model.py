
# BADSA 2025 Capstone-Project Group 3.
# Model used to train: MobileNetV2
# Dataset: HAM10000

# Here we load in all libraries we need
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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# Before we start working on the code for the model we first need to make sure our Macs don't blow up while training, so we adjust some settings here
# Configure TensorFlow for readability and sensory overload management
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show warnings only. Otherwise we would be getting spammed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations. Disabled because on Mac it can cause issues
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Reduce gRPC logging. Otherwise we would be getting spammed
os.environ['GRPC_TRACE'] = ''  # Disable gRPC tracing. This way we only see the errors (If they arise)

# We mainly code on Mac computers hence it is necessary to force TensorFlow to use CPU only otherwise our Macs might get fried
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'


# Limit CPU threads to avoid lock issues. Otherwise our Macs would get fried again
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure some settings to increase computational power
IMG_SIZE = 224 # This image size is commonly used in pre-trained ImageNet models
BATCH_SIZE = 16  # Reduced from 32 to save memory as 32 often crashed on Mac
EPOCHS = 15  # Significantly improves computational performance. We already have a long training time
FINE_TUNE_EPOCHS = 5  # Same reason as above. With more fine-rune-epochs more computational power is needed
LEARNING_RATE = 0.0001 # Safe, stable, and standard for transfer learning. Is more about training stability

# These are the class-names of our dataset
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

# Here we start to introduce the dataset and do some preprocessing
# Load the dataset
print("\nLoading metadata")
metadata = pd.read_csv('data/HAM10000_metadata.csv')
print(f"Total samples: {len(metadata)}")
print(f"\nClass distribution:")
print(metadata['dx'].value_counts())

# Create image paths because we have 2 folders where the images are in. Thus we need this function that searches both directories.
# This lets us link every metadata entry to the correct image file, no matter which directory it is from.
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

# This here created a new column called "path", where each row contains the file path corresponding to its image
metadata['path'] = metadata['image_id'].apply(get_image_path)
metadata = metadata.dropna(subset=['path']) # Here we drop any entries where no matching image was found
print(f"Valid images found: {len(metadata)}")

# Split data
print("\nSplitting data into train/validation/test sets")
train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42, stratify=metadata['dx']) # The temp_df is next used to split into the validation and test sample
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx']) # Both are equally split. So they both total 15% of the total Dataset

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Calculate class weights for imbalanced dataset. We see that there are classes that turn up more in the dataset and others that are quite rare.
# Because the model sees mostly dominant classes, it tend to ignore the rare ones. Thus we have to correct this
class_weights_values = class_weight.compute_class_weight(
    'balanced', # This computes weights so that rare classes get higher weight and common classes get lower weights
    classes=np.unique(train_df['dx']),
    y=train_df['dx']
)
class_weights_dict = dict(enumerate(class_weights_values)) # Convert the class weights as a dictionary
print("\nClass weights calculated for imbalanced dataset")

# Data augmentation for training. Augmentation increases the dataset diversity and helps prevent overfitting
print("\nSetting up data augmentation")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, # Randomly rotates images by up to +- 20 degrees
    width_shift_range=0.2,
    height_shift_range=0.2, # These two lines shift the images left/right or up/down by up to 20% of its dimension
    horizontal_flip=True,
    vertical_flip=True, # These two lines flip the images left/right or up/down
    zoom_range=0.2, # Randomly zoomes in/out by +- 20%
    fill_mode='nearest' # When the new image is created there might be empty pixels which this line tells the model to fill with the pixels from the nearby pixels
)

# Only rescaling for validation and test
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) # The images of the test and validation set should not be augmented as we want true, unmodifies images for evaluating the model

# Create data generators. As we see only on the train-set augmentation is applied.
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Here no augmentation is being made as we can see
val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='dx',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Here also so augmentation
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

# Now we finally start with building the model
# Build model with MobileNetV2
print("\nBuilding MobileNetV2 model with Transfer Learning")

# Load pre-trained MobileNetV2 (without top layer)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, # We remove the top layer as this was the original classification head which are different for us. Ours are the different skin lesion classes
    weights='imagenet'
)

# Freeze base model layers for transfer learning. We do this so the pre-trained model does no forget the useful features it already trained and so we dont destroy these learned filters
# The different low-level features are universal across image identification. Freezing also speeds up the training significantly
base_model.trainable = False
print(f"Base model loaded: {len(base_model.layers)} layers frozen")

# Build complete model
model = keras.Sequential([
    base_model, # The base model that was freezed above
    layers.GlobalAveragePooling2D(), # Converts the 2D feature maps into single 1D vector per image. Reduces parameters
    layers.Dropout(0.5), # Randomly sets 50% of the neurons to 0 during training. Helps woth overfitting
    layers.Dense(256, activation='relu'), # Fully connected layer with 256 neurons. Learns nonlinear combinations. relu keeps training fast and stable
    layers.BatchNormalization(), # Normalizes activations form the dense layer and also stabilized training
    layers.Dropout(0.3), # Another dropoutlayer after batch normalization
    layers.Dense(len(CLASS_NAMES), activation='softmax') # Final output layer. Softmay activation converts logits to probabilities for each class
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), # Adam optimizer allows for adaptive learning rates and faster convergence
    loss='categorical_crossentropy', # This is standard for multi-class classification such as ours
    metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(), keras.metrics.Recall()]
) # These different metrics are very important as accuracy alone is misleading due to the imbalance in the dataset

print("\nModel architecture:")
model.summary()

# Here we make callbacks to see how the different epochs are doing and in the end save the best model. We also tell the model what to do when loss is not improving
print("\nSetting up training callbacks")
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/best_model.h5', # The best model is saved under this directory
        monitor='val_recall', # This monitors the calidation recall at the end of every epoch
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5, # If the validation loss does not improve for 5 epochs the training stops early
        restore_best_weights=True, # After stopping, the model automatically reverts to the best weights seen during the training
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3, # If the validation loss stops improving for 3 epochs, the learning rate is reduced by factor 0.5
        min_lr=1e-7, # This is the lowest learning rate we allow in our model. It cannot go lower than this threshold
        verbose=1
    )
]

# Create models directory in order for us to save the model. Otherwise an error would occur
os.makedirs('models', exist_ok=True)

# After building the model we start with training it. This is where the magic happens
# Train model
print("\nTraining model")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}") # Logs the key parameters being used at the moment

# Fitting the model
history = model.fit(
    train_generator, # This provides the images to train on we also augmented before
    validation_data=val_generator, # This supplies the validation set which is not augmented to evaluate the performance
    epochs=EPOCHS, # This is the maximum number of passes over the entire training set. Remember the callbacks we did before can result in early stopping
    callbacks=callbacks, # This is where the callbacks come into play
    class_weight=class_weights_dict, # This adjusts the loss so that rare classes contribute more. Remember we accounted for that in the preprocessing
    verbose=1 # Displays progress bar, loss, and metrics per epoche in the console
)

# Fine-tuning: Unfreeze last layers
print("\nFine-tuning: Unfreezing last layers")
base_model.trainable = True # Setting this to true allows weights in the base model to update during the training. Fine-tuning lets the pretrained convolutional layers adapt slightly to our specific medial images

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False # Only the last 20 layers of MobileNetV2 are trainable. Low-level features are already useful in general whereas the high-level features can adapt to our dataset

print(f"Unfrozen last 20 layers for fine-tuning")

# Recompile with lower learning rate. Fine-tuning requires much smaller learning rate than the initial training of the classifier head. This prevents large updates that could destroy pretrained weights
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.Precision(), keras.metrics.Recall()]
)

# Continue training with fine-tuning
print("Fine-tuning model") # Now it trains the model again, but now the last 20 layers of the base model are unfrozen
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS, # Only a few epochs are used because fine-tuning is usually faster and more sensitive
    callbacks=callbacks,
    class_weight=class_weights_dict,
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set")
test_results = model.evaluate(test_generator, verbose=1) # Runs the trained model on the unseen test dataset which is also not augmented
print("\nTest Results:")
print(f"Loss: {test_results[0]:.4f}") # Shows how well predicitons match the true labels
print(f"Accuracy: {test_results[1]:.4f}") # Basic measure of performance
print(f"AUC: {test_results[2]:.4f}") # Measures ability to distinguish classes, robust for imbaanced data
print(f"Precision: {test_results[3]:.4f}") # This is important for when false positives are costly
print(f"Recall: {test_results[4]:.4f}") # This is more important for us as this shows the correct positive predictions over all actual positive predictions

# Save final model
model.save('models/final_model.h5')
print("\nModel saved to 'models/final_model.h5'")

# Save class names
import json
with open('models/class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f, indent=2)
print("Class names saved to 'models/class_names.json'")

# Last but not least we plot some results
# Plot training history
print("\nGenerating training plots")
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