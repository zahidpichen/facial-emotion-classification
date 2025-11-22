import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

# === CONFIGURATION ===
DATA_DIR = 'data/'
IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller batch for better generalization
EPOCHS = 50
INITIAL_EPOCHS = 20

# === 1. PARSE FILENAMES ===
label_map = {'HA': 'Happy', 'SA': 'Sad', 'AN': 'Angry', 'NE': 'Neutral', 
             'FE': 'Fear', 'DI': 'Disgust', 'SU': 'Surprise'}

filenames, categories = [], []
for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        try:
            parts = filename.split('-')
            code = parts[2]
            if code in label_map:
                filenames.append(filename)
                categories.append(label_map[code])
        except:
            pass

df = pd.DataFrame({'filename': filenames, 'category': categories})
print(f"Total images found: {len(df)}")
print(f"Class distribution:\n{df['category'].value_counts()}")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

# === 2. ADVANCED DATA AUGMENTATION ===
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# More aggressive augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df, DATA_DIR, x_col='filename', y_col='category',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
    class_mode='categorical', shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_df, DATA_DIR, x_col='filename', y_col='category',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
    class_mode='categorical', shuffle=False
)

# === 3. BUILD MODEL WITH BETTER ARCHITECTURE ===
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Initially freeze all layers
base_model.trainable = False

# Build custom head with more capacity
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(7, activation='softmax', name='emotion_output')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === 4. PHASE 1: TRAIN TOP LAYERS ===
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*60)
print("PHASE 1: Training custom head layers...")
print("="*60)

callbacks_phase1 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint('resnet_emotion_phase1.h5', monitor='val_accuracy', save_best_only=True)
]

history1 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_phase1
)

# === 5. PHASE 2: FINE-TUNE TOP LAYERS OF RESNET ===
print("\n" + "="*60)
print("PHASE 2: Fine-tuning top ResNet layers...")
print("="*60)

# Unfreeze the last 30 layers of ResNet
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Much lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Total layers: {len(model.layers)}")
print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

callbacks_phase2 = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8),
    ModelCheckpoint('resnet_emotion_model.h5', monitor='val_accuracy', save_best_only=True)
]

history2 = model.fit(
    train_generator,
    epochs=EPOCHS - INITIAL_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_phase2
)

# === 6. SAVE FINAL MODEL ===
model.save('resnet_emotion_model.h5')
print("\n✅ Final model saved as 'resnet_emotion_model.h5'")

# === 7. EVALUATION ===
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Print class indices for reference
print("\nClass Indices:")
for emotion, idx in train_generator.class_indices.items():
    print(f"{idx}: {emotion}")
