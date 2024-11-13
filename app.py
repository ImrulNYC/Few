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
            return render_template('index.html', filename=filename, label=label)

    # For GET requests, reset values
    return render_template('index.html', filename=None, label=None)


if __name__ == "__main__":
    # Use the PORT environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)