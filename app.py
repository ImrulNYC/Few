from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from prediction import predict_flower

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

from datetime import datetime

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    label = None

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

            # Call the helper function to predict
            predicted_label, confidence = predict_flower(file_path)
            if predicted_label:
                label = f"Predicted Flower: {predicted_label} with {confidence:.2f}% confidence."
            else:
                label = "The flower cannot be confidently recognized. Please try another image."

            # Pass filename and label to the template
            return render_template('index.html', filename=filename, label=label, cache_buster=datetime.now().timestamp())

    # For GET requests, reset values
    return render_template('index.html', filename=None, label=None)




@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename=f'uploads/{filename}')

if __name__ == '__main__':
    app.run(debug=True)
