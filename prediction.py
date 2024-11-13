import os
import requests
from tensorflow.keras.models import load_model
import json


# Configuration
MODEL_URL = "https://flowerm.s3.us-east-1.amazonaws.com/flower_model_best.keras"
MODEL_PATH = "flower_model_best.keras"

# Ensure the model exists locally
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()  # Raise exception if the request fails
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded successfully to {MODEL_PATH}.")

# Call the function to ensure the model exists before loading
ensure_model_exists()

# Load the model globally for predictions
model = load_model(MODEL_PATH)

def predict_flower(img_path):
    """
    Predict the type of flower from the image.

    Parameters:
        img_path (str): Path to the input image.

    Returns:
        tuple: Predicted label and confidence percentage.
    """
    from tensorflow.keras.preprocessing import image
    import numpy as np

    # Image preprocessing
    img_width, img_height = 288, 276
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image

    # Predict using the loaded model
    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100  # Get confidence in percentage
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Load class labels from a file
    CLASS_LABELS_PATH = "data/class_labels.json"
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = json.load(f)

    predicted_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class)]

    # Only return the result if confidence is high enough
    if confidence >= 80:
        return predicted_label, confidence
    else:
        return None, None

