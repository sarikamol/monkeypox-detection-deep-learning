from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained MobileNet model
MODEL_PATH = os.path.join(os.getcwd(), 'monkeypox_mobilenet_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Define the class names (ensure these match your model's training labels)
class_names = ['Monkeypox', 'No Monkeypox']

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = 224  # Replace 224 with the size used during training

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    # Debugging: Log predictions and shape
    logger.debug("Predictions: %s", predictions)
    logger.debug("Predictions shape: %s", predictions.shape)

    if predictions.size == 0:
        raise ValueError("Predictions array is empty.")

    predicted_class = class_names[np.argmax(predictions[0])]  # Fix here: use predictions[0] instead of predictions[1]
    confidence = round(100 * np.max(predictions[0]), 2)  # Use predictions[0] for the confidence

    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if not allowed_file(file.filename):
        return "Unsupported file type. Please upload a PNG, JPG, or JPEG image.", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        label, confidence = predict(model, file_path)
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        return "An error occurred while making predictions. Please try again later.", 500

    return render_template('result.html', label=label, confidence=confidence, image_path=file_path)

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

if __name__ == '__main__':
    app.run(debug=True)
