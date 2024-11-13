import os
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

# Configuration for model and labels
MODEL_URL = 'https://flowerm.s3.us-east-1.amazonaws.com/flower_model_best.keras'
MODEL_PATH = "flower_model_best.keras"
CLASS_LABELS_PATH = "data/class_labels.json"
IMG_WIDTH, IMG_HEIGHT = 288, 276

# Function to download the model
def download_model(url, local_path):
    """
    Downloads a file from the given URL to a local path.

    Parameters:
        url (str): The URL of the file to download.
        local_path (str): The local path where the file should be saved.
    """
    print(f"Downloading model from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded successfully to {local_path}.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download model: {e}")

# Ensure the model is available
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# Load the trained model
model = load_model(MODEL_PATH)

# Load class labels
if not os.path.exists(CLASS_LABELS_PATH):
    raise FileNotFoundError(f"Class labels file not found: {CLASS_LABELS_PATH}")
with open(CLASS_LABELS_PATH, 'r') as f:
    class_labels = json.load(f)

def predict_flower(img_path):
    """
    Predicts the flower type from an image.

    Parameters:
        img_path (str): Path to the image.

    Returns:
        tuple: (predicted_label, confidence_percentage) or (None, None) if confidence is too low.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # Preprocess the image
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image

    # Predict the flower
    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100  # Convert to percentage
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get the label of the predicted class
    predicted_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class)]

    # Return prediction only if confidence is >= 80%
    if confidence >= 80:
        return predicted_label, confidence
    else:
        return None, None
