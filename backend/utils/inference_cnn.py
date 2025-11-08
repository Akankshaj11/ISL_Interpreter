import tensorflow as tf
import numpy as np
import json
import os

# Load model and label map
MODEL_PATH = os.path.join("models", "cnn_model.h5")
LABEL_MAP_PATH = os.path.join("models", "label_map.json")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
except Exception as e:
    print("Error loading model or label map:", e)
    model = None
    label_map = {}

def predict_static(landmarks):
    """
    Predict the static sign (A-Z, 0-9) using CNN model.
    landmarks: numpy array of shape (63,)  -> 21 landmarks × 3 coordinates
    """
    if model is None:
        return "Model not loaded"

    # reshape input to match model
    input_data = np.expand_dims(landmarks, axis=0)
    predictions = model.predict(input_data, verbose=0)
    class_id = int(np.argmax(predictions))
    
    # Get label name from JSON map
    label_name = label_map.get(str(class_id), "Unknown")
    return label_name
