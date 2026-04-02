"""
FitOS — model.py (Fixed)
Fixes from your original:
  ✓ Missing `import tensorflow as tf` added
  ✓ Safer class loading with fallback
  ✓ class_map.json used if dataset folder not present (for deployment)
  ✓ Handles RGBA images correctly
  ✓ Returns confidence as 0–1 float (app.py multiplies by 100)
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "phase2_best.h5")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_map.json")
DATASET_DIR   = os.path.join(BASE_DIR, "../Indian Food Images/Indian Food Images")

# ── Load class list ──────────────────────────────────────
# Priority: class_map.json (always present after training)
# Fallback: scan dataset directory
classes = []

if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH) as f:
        class_map = json.load(f)
    # class_map is {name: index} — invert to {index: name}
    classes = [None] * len(class_map)
    for name, idx in class_map.items():
        classes[idx] = name
    print(f"Loaded {len(classes)} classes from class_map.json")

elif os.path.exists(DATASET_DIR):
    classes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and
        len(os.listdir(os.path.join(DATASET_DIR, d))) > 0
    ])
    print(f"Loaded {len(classes)} classes from dataset directory")

else:
    print("WARNING: No class map or dataset found. Predictions will return 'unknown'.")

print(f"Sample classes: {classes[:5]}")

# ── Load Keras model ─────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
else:
    print(f"WARNING: Model not found at {MODEL_PATH}")
    print("Train the model first using train_model.py")


# ── Prediction function ──────────────────────────────────
def predict_food(image: Image.Image):
    """
    Args:
        image: PIL Image object
    Returns:
        (food_name: str, confidence: float)  — confidence is 0.0–1.0
    """
    if model is None or not classes:
        return "unknown", 0.0

    # Resize and convert to RGB (handles RGBA, grayscale, etc.)
    image = image.convert("RGB").resize((224, 224))

    # Normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    predictions = model.predict(img_array, verbose=0)

    class_index = int(np.argmax(predictions[0]))
    confidence  = float(np.max(predictions[0]))

    if class_index >= len(classes):
        return "unknown", confidence

    food_name = classes[class_index]
    return food_name, confidence