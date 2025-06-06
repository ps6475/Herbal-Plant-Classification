import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
from PIL import Image

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 80  # Set this based on your dataset

# Load MobileNetV2 base model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Unfreeze last 40 layers
for layer in base_model.layers[-40:]:
    layer.trainable = True

# Build the model
x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compil(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # Label smoothing added
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_datagen.flow_from_directory(
    "C:\\Users\\HP\\Downloads\\SEAI PROJECT\\Medicinal Leaf dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    "C:\\Users\\HP\\Downloads\\SEAI PROJECT\\Medicinal Leaf dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# Reverse class labels
class_indices = train_data.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Train the model
history = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=[lr_scheduler, early_stopping]
)

# Evaluate the model
val_loss, val_accuracy, top3_accuracy = model.evaluate(val_data)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Top-3 Accuracy: {top3_accuracy:.4f}")

# Save model
model.save("herbal_plant_classifier_v3_improved.h5")

# Predict function
def predict_plant(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_plant = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_plant, confidence

# Test prediction
sample_image_path = "C:\\Users\\HP\\Downloads\\SEAI PROJECT\\Medicinal Leaf dataset\\Aloevera\\2.jpg"
try:
    plant_name, confidence = predict_plant(sample_image_path)
    print(f"Predicted Plant: {plant_name}, Confidence: {confidence:.4f}")
except Exception as e:
    print(f"Error predicting image: {e}")

print("✅ Training and prediction completed with improved config!")
