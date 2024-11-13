import os
import json
import numpy as np
import gdown
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'supersecretkey'  # Needed for flash messaging

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Google Drive file ID for the model
file_id = '1uTzxTMij9s_YFv3lAVY6sqwP2diCRm0A'
model_path = 'flower_model_final.keras'

# Check if the model file exists; if not, download it
if not os.path.exists(model_path):
    url = 'https://drive.google.com/file/d/1uTzxTMij9s_YFv3lAVY6sqwP2diCRm0A/view?usp=share_link'
    gdown.download(url, model_path, quiet=False)

# Load the trained model
model = load_model(model_path)

# Load class labels from the JSON file
with open(os.path.join('data', 'class_labels.json'), 'r') as f:
    class_labels = json.load(f)

# Image dimensions
img_width, img_height = 288, 276

# Function to predict the flower type for an uploaded image
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = list(class_labels.keys())[list(class_labels.values()).index(predicted_class)]

    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict the flower type
            predicted_label = predict_flower(file_path)
            flash(f'Predicted Flower: {predicted_label}')

            return render_template('index.html', filename=filename, label=predicted_label)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename=f'uploads/{filename}')

if __name__ == '__main__':
    app.run(debug=True)
