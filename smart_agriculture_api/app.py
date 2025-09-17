import base64
import os
import io
import joblib
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Flask app
app = Flask(__name__)
# Enable CORS for your frontend domain
CORS(app, resources={r"/*": {"origins": ["https://smart-agriculture-system-delta.vercel.app"]}})

# Load Models
irrigation_model, scaler = joblib.load("models/irrigation_model.pkl")

# Load a smaller MobileNetV2 for memory optimization
base_model = MobileNetV2(weights='imagenet', include_top=False, alpha=0.35, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
prediction = Dense(15, activation='softmax')(x)
plant_model = Model(inputs=base_model.input, outputs=prediction)
plant_model.load_weights("models/plant_disease_model.h5")

# Class metadata (causes, symptoms, treatments)
class_names = [
    "Pepper Bell - Bacterial Spot", "Pepper Bell - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", 
    "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus", "Tomato - Healthy"
]

causes = {...}  # same as before
symptoms = {...}
treatments = {...}

# Helper: preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Weather route
@app.route("/check_weather", methods=["GET"])
def check_weather():
    location = request.args.get("location", "default_location")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        rain_forecast = "rain" in weather_data["weather"][0]["description"].lower()
        altitude = weather_data["coord"]["lat"] * 0.1
        return jsonify({
            "rain_expected": rain_forecast,
            "temperature": weather_data["main"]["temp"],
            "pressure": weather_data["main"]["pressure"],
            "altitude": altitude
        })
    return jsonify({"error": "Could not fetch weather data"}), 400

# Irrigation prediction
@app.route("/predict/irrigation", methods=["POST"])
def predict_irrigation():
    try:
        data = request.json
        required_fields = ["temperature", "soil_moisture", "pressure", "altitude"]
        if not all(k in data for k in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        X = np.array([[float(data["temperature"]),
                       float(data["pressure"]),
                       float(data["altitude"]),
                       float(data["soil_moisture"])]])
        X_scaled = scaler.transform(X)
        pred = irrigation_model.predict(X_scaled)[0]

        advice = ""
        if pred in [0, 1]:
            if data["temperature"] > 35:
                advice = "High temperature detected, irrigation is strongly recommended."
            elif data["pressure"] < 1000:
                advice = "Low atmospheric pressure detected. Irrigation is advised."
            else:
                advice = "Soil moisture is low, irrigation is recommended."
        else:
            advice = "No irrigation needed."

        return jsonify({"prediction": advice})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Plant disease prediction
@app.route("/predict/plant", methods=["POST"])
def predict_plant_disease():
    try:
        if "image" in request.files:
            file = request.files["image"]
            img = Image.open(file).convert("RGB")
        elif "image" in request.json:
            image_data = request.json["image"].split(",")[1]
            img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        else:
            return jsonify({"error": "No image provided"}), 400

        img_array = preprocess_image(img)
        prediction = plant_model.predict(img_array)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction)) * 100
        predicted_class = class_names[predicted_class_idx]

        if "Healthy" in predicted_class:
            return jsonify({"healthy": "Plant is healthy", "confidence": confidence})

        return jsonify({
            "disease": predicted_class,
            "confidence": confidence,
            "cause": causes.get(predicted_class, "N/A"),
            "symptoms": symptoms.get(predicted_class, "N/A"),
            "treatment": treatments.get(predicted_class, "N/A")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
