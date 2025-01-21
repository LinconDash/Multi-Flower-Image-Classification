from flask import Flask, render_template, request, jsonify

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
from src.components.data_loader import DataGenerator

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "models/vgg16_flower_classifier.h5"
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Define class labels
CLASS_NAMES = os.listdir("artifacts/train")


def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess the uploaded image to the required format for prediction.
    
    Args:
        image (PIL.Image.Image): Uploaded image.
        target_size (tuple): Target size for resizing the image.
    
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the file is part of the request
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded!"})
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file!"})
        
        try:
            # Open and preprocess the image
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)

            # Perform prediction
            predictions = model.predict(processed_image)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100

            if confidence >= 80:
                sentence = f"It's definitely {predicted_class}, with a confidence of ({confidence:.2f}%)"
            elif confidence >= 60:
                sentence = f"I think it is {predicted_class}, with a confidence of ({confidence:.2f}%)" 
            elif confidence >= 40:
                sentence = f"Maybe it is {predicted_class}, with a confidence of ({confidence:.2f}%)"
            else:
                sentence = f"I don't know, is it a {predicted_class} ? , with a confidence of ({confidence:.2f}%)"
            # Return the result to the HTML page
            return render_template(
                "index.html",
                prediction=sentence,
            )
        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)