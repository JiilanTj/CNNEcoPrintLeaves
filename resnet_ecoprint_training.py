import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
from datetime import datetime

# Konfigurasi GPU/CPU
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Konfigurasi parameter
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100

# Data augmentation yang lebih ringan
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
)

# Load dataset
dataset_path = './dataset'
print("Daftar folder dalam dataset:")
folders = sorted(os.listdir(dataset_path))
print(folders)

# Karakteristik daun untuk setiap kelas
leaf_characteristics = {
    'bambujapan': {
        'tekstur': 'halus',
        'urat': 'sejajar, tidak menonjol',
        'kandungan_air': 'sedang'
    },
    'binahong': {
        'tekstur': 'halus',
        'urat': 'melengkung, tidak menonjol',
        'kandungan_air': 'tinggi'
    },
    'bodhi': {
        'tekstur': 'kasar',
        'urat': 'menonjol',
        'kandungan_air': 'rendah'
    },
    'jarakmerah': {
        'tekstur': 'kasar',
        'urat': 'menonjol',
        'kandungan_air': 'rendah'
    },
    'jati': {
        'tekstur': 'sangat kasar',
        'urat': 'sangat menonjol',
        'kandungan_air': 'rendah'
    },
    'kamboja': {
        'tekstur': 'halus',
        'urat': 'tidak menonjol',
        'kandungan_air': 'tinggi'
    },
    'kayuafrika': {
        'tekstur': 'kasar',
        'urat': 'menonjol',
        'kandungan_air': 'rendah'
    },
    'lanang': {
        'tekstur': 'kasar',
        'urat': 'menonjol',
        'kandungan_air': 'rendah'
    },
    'palemjamrud': {
        'tekstur': 'halus',
        'urat': 'tidak menonjol',
        'kandungan_air': 'sedang'
    },
    'tulak': {
        'tekstur': 'halus',
        'urat': 'tidak menonjol',
        'kandungan_air': 'tinggi'
    }
}

# Load dan preprocess dataset
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Model architecture
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Arsitektur yang lebih sederhana
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Learning rate yang lebih tinggi untuk awal
optimizer = Adam(learning_rate=0.001)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    mode='max'
)

# Learning rate reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    mode='max',
    verbose=1
)

# Model checkpoint
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_resnet.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Training
try:
    print("\nMemulai training fase pertama...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    print("\nMemulai fine-tuning...")
    # Unfreeze beberapa layer terakhir untuk fine-tuning
    for layer in base_model.layers[-30:]:  # Unfreeze 30 layer terakhir
        layer.trainable = True
    
    # Compile ulang dengan learning rate yang lebih kecil
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tuning
    history_fine = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Plot hasil
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results_resnet.png')
    plt.show()
    
    # Simpan history training
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history_resnet.csv', index=False)
    
    # Simpan model
    model.save('model_daun_ecoprint_resnet.h5')
    print("\nModel dan hasil training berhasil disimpan!")

except Exception as e:
    print(f"Error during training: {str(e)}")