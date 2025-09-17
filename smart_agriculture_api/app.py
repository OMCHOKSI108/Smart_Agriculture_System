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

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# Force CPU to avoid GPU errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Load Models
irrigation_model, scaler = joblib.load("models/irrigation_model.pkl")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
prediction = Dense(15, activation='softmax')(x)
plant_model = Model(inputs=base_model.input, outputs=prediction)
plant_model.load_weights("models/plant_disease_model.h5")

# Weather API Key 
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Image Preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array


class_names = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy"
]

causes = {
    "Pepper Bell - Bacterial Spot": "Xanthomonas campestris bacteria",
    "Potato - Early Blight": "Alternaria solani fungus",
    "Potato - Late Blight": "Phytophthora infestans pathogen",
    "Tomato - Bacterial Spot": "Xanthomonas campestris bacteria",
    "Tomato - Early Blight": "Alternaria solani fungus",
    "Tomato - Late Blight": "Phytophthora infestans pathogen",
    "Tomato - Leaf Mold": "Passalora fulva fungus",
    "Tomato - Septoria Leaf Spot": "Septoria lycopersici fungus",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)": "Tetranychus urticae mites",
    "Tomato - Target Spot": "Corynespora cassiicola fungus",
    "Tomato - Yellow Leaf Curl Virus": "Tomato yellow leaf curl virus (spread by whiteflies)",
    "Tomato - Mosaic Virus": "Tobacco mosaic virus (TMV)",
}

symptoms = {
    "Pepper Bell - Bacterial Spot": "Small, dark, water-soaked spots on leaves and fruit; can lead to defoliation and reduced yield.",
    "Potato - Early Blight": "Brown lesions with concentric rings on lower leaves, leading to defoliation.",
    "Potato - Late Blight": "Dark, water-soaked lesions on leaves and stems; rapid plant death in humid conditions.",
    "Tomato - Bacterial Spot": "Dark, water-soaked leaf spots that merge and cause yellowing; fruit may have scabby spots.",
    "Tomato - Early Blight": "Dark spots with concentric rings on lower leaves; can cause plant defoliation.",
    "Tomato - Late Blight": "Large, water-soaked lesions on leaves and stems, leading to rapid plant collapse.",
    "Tomato - Leaf Mold": "Yellow patches on upper leaves, fuzzy olive-green mold underneath.",
    "Tomato - Septoria Leaf Spot": "Numerous small, circular spots with dark margins on leaves; causes defoliation.",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)": "Yellow speckling on leaves, fine webbing, leaf curling, and drying.",
    "Tomato - Target Spot": "Circular brown spots with concentric rings, leading to premature leaf drop.",
    "Tomato - Yellow Leaf Curl Virus": "Upward curling, yellowing of leaves, and stunted growth.",
    "Tomato - Mosaic Virus": "Mottled yellow and green leaf pattern, distorted leaf growth, and reduced fruit production.",
}


treatments = {
    "Pepper Bell - Bacterial Spot": "Apply copper-based fungicides. Avoid overhead watering. Use disease-resistant varieties.",
    "Pepper Bell - Healthy": "No treatment needed. Maintain proper watering and nutrient balance.",
    "Potato - Early Blight": "Use fungicides containing chlorothalonil or mancozeb. Rotate crops and remove infected plants.",
    "Potato - Late Blight": "Apply fungicides like metalaxyl or chlorothalonil. Improve air circulation and avoid wet conditions.",
    "Potato - Healthy": "No treatment needed. Maintain soil health and proper irrigation.",
    "Tomato - Bacterial Spot": "Use copper sprays and bactericides. Avoid handling plants when wet.",
    "Tomato - Early Blight": "Apply fungicides such as chlorothalonil or mancozeb. Prune lower leaves and use mulch to prevent soil splash.",
    "Tomato - Late Blight": "Use fungicides like copper-based sprays or chlorothalonil. Remove and destroy infected plants.",
    "Tomato - Leaf Mold": "Increase air circulation, reduce humidity, and apply fungicides containing chlorothalonil or mancozeb.",
    "Tomato - Septoria Leaf Spot": "Use fungicides like copper-based sprays. Remove infected leaves and ensure good airflow.",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)": "Use insecticidal soap, neem oil, or predatory mites to control infestations.",
    "Tomato - Target Spot": "Apply fungicides such as azoxystrobin or chlorothalonil. Remove infected leaves and avoid overhead watering.",
    "Tomato - Yellow Leaf Curl Virus": "Control whiteflies with neem oil or insecticidal soap. Use virus-resistant tomato varieties.",
    "Tomato - Mosaic Virus": "Remove infected plants immediately. Control aphids that spread the virus using neem oil or insecticidal soap.",
    "Tomato - Healthy": "No treatment needed. Maintain good growing conditions."
}


# API Routes 
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

@app.route("/predict/irrigation", methods=["POST"])
def predict_irrigation():
    try:
        data = request.json
        required_fields = ["temperature", "soil_moisture", "pressure", "altitude"]
        if not all(k in data for k in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Convert input values
        temp = float(data["temperature"])
        soil = float(data["soil_moisture"])
        pressure = float(data["pressure"])
        altitude = float(data["altitude"])


        X = np.array([[temp, pressure, altitude, soil]])
        X_scaled = scaler.transform(X)

        prediction = irrigation_model.predict(X_scaled)[0]

        if prediction in [0, 1]: 
            if temp > 35:
                advice = "High temperature detected, irrigation is strongly recommended to prevent heat stress."
            elif pressure < 1000:
                advice = "Low atmospheric pressure detected, indicating possible weather changes. Irrigation is advised."
            else:
                advice = "Soil moisture is low, irrigation is recommended."

        elif prediction in [2, 3]: 
            if temp < 15:
                advice = "Low temperature detected, moisture evaporation is slow. No irrigation needed."
            elif pressure > 1020:
                advice = "High atmospheric pressure detected, reducing evaporation. No irrigation required."
            else:
                advice = "Soil is sufficiently wet, irrigation is not required."

        else:
            advice = "Unexpected model output. Please verify input data."


        return jsonify({"prediction": advice})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        if predicted_class == "Pepper Bell - Healthy" or predicted_class == "Tomato - Healthy" or predicted_class == "Potato - Healthy": 
            return jsonify({"healthy": "Plant is healthy", "confidence": confidence})
        cause = causes.get(predicted_class, "Cause details not available")
        symptom = symptoms.get(predicted_class, "Symptom details not available")
        treatment = treatments.get(predicted_class, "No treatment details available")


        return jsonify({
            "disease": predicted_class,
            "confidence": confidence,
            "cause": cause,
            "symptoms": symptom,
            "treatment": treatment
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
