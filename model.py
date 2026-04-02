import os
import json
import numpy as np
from PIL import Image

# SAFE import (prevents crash)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import error:", e)
    TF_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "phase2_best.h5")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_map.json")

# ── Load classes ─────────────────────────────
classes = []

if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH) as f:
        class_map = json.load(f)

    # SORT FIX (IMPORTANT)
    classes = [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]

    print(f"Loaded {len(classes)} classes")

else:
    print("WARNING: class_map.json not found")

# ── Load model ─────────────────────────────
model = None

if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Model load error:", e)
else:
    print("⚠️ TensorFlow not available or model missing")

# ── Prediction ─────────────────────────────
def predict_food(image: Image.Image):
    if model is None or not classes:
        return "unknown", 0.0

    image = image.convert("RGB").resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    class_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    if class_index >= len(classes):
        return "unknown", confidence

    return classes[class_index], confidence