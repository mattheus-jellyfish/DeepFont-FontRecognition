"""
# Single Font Detection Model
A complete implementation for training a binary font classifier to detect a specific target font.
Can be run both locally and on Google Colab Enterprise.

## Requirements:
- tensorflow
- PIL
- numpy
- matplotlib
- trdg (Text Recognition Data Generator)
- fonttools

## Usage:
1. Set TARGET_FONT_PATH to your .otf font file
2. Adjust hyperparameters if needed
3. Run locally or in Colab Enterprise
"""

# Import required libraries
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.font_manager as fm

# Configuration
class Config:
    # Paths
    TARGET_FONT_PATH = os.path.join(os.path.dirname(__file__), "Denton-Light.otf")  # Path to target font relative to script location
    OUTPUT_DIR = "binary_dataset"
    MODEL_PATH = "binary_font_detector.h5"
    
    # Training parameters
    SAMPLES_PER_CLASS = 2000
    IMAGE_SIZE = 105
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    
    # Data generation
    NUM_NEGATIVE_FONTS = 30  # Number of different fonts to use for negative samples
    TEXT_COLORS = "#000000,#4A4A4A,#666666"
    
    # Model parameters
    DROPOUT_RATE = 0.5
    
    # Inference
    CONFIDENCE_THRESHOLD = 0.5
    PATCH_OVERLAP = 0.5  # 50% overlap between patches

# Data Generation Functions
def setup_directories():
    """Create necessary directories for dataset"""
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "target_font"), exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "other_fonts"), exist_ok=True)
    print("Created dataset directories")

def generate_synthetic_data():
    """Generate synthetic training data using TRDG"""
    print("Starting synthetic data generation...")
    
    # Generate target font samples
    target_cmd = (
        f"trdg --output_dir \"{os.path.join(Config.OUTPUT_DIR, 'target_font')}\" "
        f"-c {Config.SAMPLES_PER_CLASS} -b 3 -bl 3 -rbl -k 6 -rk "
        f"-tc {Config.TEXT_COLORS} -f {Config.IMAGE_SIZE} -cs 3 "
        f"-ft \"{Config.TARGET_FONT_PATH}\""
    )
    os.system(target_cmd)
    print("Generated target font samples")
    
    # Find and use system fonts for negative samples
    system_fonts = [f.path for f in fm.fontManager.ttflist 
                   if f.path.endswith(('.ttf', '.otf'))]
    print(f"Found {len(system_fonts)} system fonts")
    
    samples_per_font = Config.SAMPLES_PER_CLASS // min(len(system_fonts), 
                                                     Config.NUM_NEGATIVE_FONTS)
    
    for i, font_path in enumerate(system_fonts[:Config.NUM_NEGATIVE_FONTS]):
        print(f"Generating samples for negative font {i+1}/{Config.NUM_NEGATIVE_FONTS}")
        other_cmd = (
            f"trdg --output_dir \"{os.path.join(Config.OUTPUT_DIR, 'other_fonts')}\" "
            f"-c {samples_per_font} -b 3 -bl 3 -rbl -k 6 -rk "
            f"-tc {Config.TEXT_COLORS} -f {Config.IMAGE_SIZE} -cs 3 "
            f"-ft \"{font_path}\""
        )
        os.system(other_cmd)
    
    print("Completed synthetic data generation")

# Image Processing Functions
def add_noise(image):
    """Add random noise to image for data augmentation"""
    img_array = np.array(image)
    noise = np.random.normal(0, 5, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def preprocess_dataset():
    """Apply preprocessing to all generated images"""
    print("Starting dataset preprocessing...")
    
    for class_dir in ["target_font", "other_fonts"]:
        dir_path = os.path.join(Config.OUTPUT_DIR, class_dir)
        for file in os.listdir(dir_path):
            if file.endswith(('.jpg', '.png')):
                file_path = os.path.join(dir_path, file)
                try:
                    # Load, convert to grayscale, add noise, and save
                    img = Image.open(file_path).convert('L')
                    noisy_img = add_noise(img)
                    noisy_img.save(file_path)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
    
    print("Completed dataset preprocessing")

# Model Definition
def create_model():
    """Create the binary classification model"""
    model = Sequential([
        # Feature extraction
        Conv2D(64, (3, 3), activation='relu', padding='same', 
               input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Classification
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(Config.DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Training Functions
def train_model():
    """Train the font detection model"""
    print("Starting model training...")
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1
    )
    
    train_generator = train_datagen.flow_from_directory(
        Config.OUTPUT_DIR,
        target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        Config.OUTPUT_DIR,
        target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        subset='validation'
    )
    
    # Create and train model
    model = create_model()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=0.00001
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )
    
    # Save model
    model.save(Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")
    
    return history

def plot_training_history(history):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Inference Functions
def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    img = Image.open(image_path).convert('L')
    
    # Resize maintaining aspect ratio
    baseheight = Config.IMAGE_SIZE
    hpercent = (baseheight/float(img.size[1]))
    wsize = int((float(img.size[0])*float(hpercent)))
    img = img.resize((wsize, baseheight), Image.LANCZOS)
    
    # Create overlapping patches
    patches = []
    step_size = int(Config.IMAGE_SIZE * (1 - Config.PATCH_OVERLAP))
    
    for i in range(0, img.width - Config.IMAGE_SIZE + 1, step_size):
        patch = img.crop((i, 0, i + Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        patches.append(np.array(patch))
    
    # Handle images smaller than patch size
    if not patches:
        patch = img.crop((0, 0, min(img.width, Config.IMAGE_SIZE), Config.IMAGE_SIZE))
        patches.append(np.array(patch))
    
    # Normalize
    patches = np.array(patches).reshape(-1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 1) / 255.0
    
    return patches

def detect_font(image_path, model_path=Config.MODEL_PATH):
    """Detect if image contains the target font"""
    # Load model
    model = load_model(model_path)
    
    # Preprocess image
    patches = preprocess_image(image_path)
    
    # Get predictions
    predictions = model.predict(patches)
    
    # Aggregate results
    max_confidence = float(max(predictions)[0]) if predictions.size > 0 else 0
    is_target_font = max_confidence > Config.CONFIDENCE_THRESHOLD
    
    return {
        "is_target_font": bool(is_target_font),
        "confidence": max_confidence,
        "num_patches": len(patches)
    }

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Check if running in Colab
    IN_COLAB = 'google.colab' in sys.modules
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        # Update paths for Colab
        Config.OUTPUT_DIR = "/content/binary_dataset"
        Config.MODEL_PATH = "/content/drive/MyDrive/binary_font_detector.h5"
    
    # Create directories and generate data
    setup_directories()
    generate_synthetic_data()
    preprocess_dataset()
    
    # Train model
    history = train_model()
    plot_training_history(history)
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()