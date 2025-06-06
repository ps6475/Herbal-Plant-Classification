import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset (same path you used during training)
dataset_path = "C:/Users/HP/Downloads/SEAI PROJECT/Medicinal Leaf dataset"

# Set image size and batch size as used during training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Load dataset to access class indices
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # no shuffle needed here
)

# Extract class indices and reverse them
class_indices = train_data.class_indices
label_mapping = {str(v): k for k, v in class_indices.items()}

# Save to JSON
json_filename = "class_labels.json"
with open(json_filename, "w", encoding="utf-8") as json_file:
    json.dump(label_mapping, json_file, indent=4)

print(f"✅ Class labels saved to {json_filename}")
