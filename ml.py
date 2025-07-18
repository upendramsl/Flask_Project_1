from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tempfile

app = Flask(__name__)

# Load the model once when the server starts
MODEL_PATH = "best_MobileNet.h5"
model = load_model(MODEL_PATH)

# Class mapping
class_indices = {
    0: 'Arborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
}
@app.route("/predict", methods=["POST"])

def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Save the uploaded image to a temporary location
        image_file = request.files['image']
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image_file.save(temp_file.name)
        print(image_file)
        # Preprocess image
        img = load_img(temp_file.name, target_size=(50, 50))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        print(prediction)

        predicted_class = class_indices[np.argmax(prediction[0])]

        print(predicted_class)
        confidence = float(np.max(prediction[0]) * 100)
        print(prediction)
        # Cleanup
        print(predicted_class)
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
