import tensorflow as tf
import numpy as np
import json
import os
from collections import deque

# Load model and label map
MODEL_PATH = os.path.join("models", "lstm_model.h5")
LABEL_MAP_PATH = os.path.join("models", "label_map.json")

# Try loading the LSTM model and label map
try:
    lstm_model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
except Exception as e:
    print("⚠️ Error loading LSTM model or label map:", e)
    lstm_model = None
    label_map = {}

# Queue to store sequence of frames (landmarks)
SEQUENCE_LENGTH = 30
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)


def update_sequence(landmarks):
    """
    Update the frame sequence buffer with new landmarks.
    landmarks: np.array of shape (63,)  -> 21 landmarks * 3 coordinates
    """
    sequence_buffer.append(landmarks)
    return len(sequence_buffer) == SEQUENCE_LENGTH


def predict_dynamic():
    """
    Predict gesture from a sequence of landmarks using LSTM.
    Returns recognized label or 'Waiting...'
    """
    if lstm_model is None or len(sequence_buffer) < SEQUENCE_LENGTH:
        return "Waiting..."

    # Prepare data for model
    input_sequence = np.expand_dims(np.array(sequence_buffer), axis=0)  # shape (1, 30, 63)

    # Predict
    predictions = lstm_model.predict(input_sequence, verbose=0)
    class_id = int(np.argmax(predictions))
    label_name = label_map.get(str(class_id), "Unknown")

    return label_name


def reset_sequence():
    """Clear the stored landmark sequence."""
    sequence_buffer.clear()
