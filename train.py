"""
Brain Tumor MRI Image Classification - Training Script
Supports Custom CNN and Transfer Learning (ResNet50 / MobileNetV2)
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "dataset" # Replace with actual path to dataset

def create_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found. Please organize data into subfolders by class.")
        return None, None
    
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    return train_ds, val_ds

def build_custom_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    return model

def build_transfer_model(num_classes, base_model_name='resnet50'):
    if base_model_name == 'resnet50':
        base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    else:
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    return model

if __name__ == "__main__":
    print("Brain Tumor Classification Training Script Initialized.")
