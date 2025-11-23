import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import mixed_precision

# === GPU CONFIGURATION ===
# Prevent TensorFlow from eating all VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Memory Growth Enabled: {len(gpus)} GPUs")
    except RuntimeError as e:
        print(e)

mixed_precision.set_global_policy('mixed_float16')

# === CONFIGURATION ===
DATA_DIR = 'data/'
IMG_SIZE = 224
BATCH_SIZE = 32  # Increased for better GPU utilization
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

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['category'])
num_classes = len(le.classes_)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

# === 2. OPTIMIZED DATA PIPELINE (tf.data) ===
from tensorflow.keras.applications.resnet_v2 import preprocess_input

def load_image(filename, label):
    # Load and decode image
    img_path = tf.strings.join([DATA_DIR, filename])
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    # Preprocess for ResNet
    img = preprocess_input(img)
    # One-hot encode label
    label = tf.one_hot(label, num_classes)
    return img, label

def create_dataset(dataframe, is_training=True):
    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((
        dataframe['filename'].values, 
        dataframe['label_encoded'].values
    ))
    
    # Parallel loading and preprocessing
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache in RAM (since dataset is small ~2500 images)
    dataset = dataset.cache()
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Batch and prefetch
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset(train_df, is_training=True)
val_ds = create_dataset(val_df, is_training=False)

# === 3. BUILD MODEL WITH GPU AUGMENTATION ===
# Augmentation layers run on GPU!
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.2),
], name="data_augmentation")

# Base model
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

# Build model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)  # Apply augmentation on GPU
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax', name='emotion_output')(x)

model = Model(inputs=inputs, outputs=outputs)

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
    train_ds,
    epochs=INITIAL_EPOCHS,
    validation_data=val_ds,
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
    train_ds,
    epochs=EPOCHS - INITIAL_EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_phase2
)

# === 6. SAVE FINAL MODEL ===
model.save('resnet_emotion_model.h5')
print("\n✅ Final model saved as 'resnet_emotion_model.h5'")

# === 7. EVALUATION ===
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Print class indices for reference
print("\nClass Indices:")
for i, emotion in enumerate(le.classes_):
    print(f"{i}: {emotion}")
