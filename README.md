# Smart Agriculture System / FarmFriend

This project implements a Smart Agriculture System using deep learning and machine learning models for plant disease detection and intelligent irrigation scheduling. It leverages MobileNetV2 for image classification and logistic regression for binary prediction.

# 📁 Project Structure

- `PlantDiseaseDetection/` - Contains code and model for detecting crop diseases using MobileNetV2
- `IrrigationScheduling/` - ML model code for predicting irrigation needs based on weather and soil data
- `smart_agriculture_api/` - Flask API for backend integration
- `smart-agriculture-frontend/` - React+Vite based frontend for user interaction

# 🧠 Models Used

## 1. MobileNetV2 – Plant Disease Detection

- Type: Deep CNN using Transfer Learning
- Layers Added: GlobalAveragePooling2D, Dense layers, BatchNormalization, Dropout, Softmax output
- Optimizer: Adam | Loss: Categorical Crossentropy
- Accuracy: ~90%
- Why MobileNetV2: Lightweight, accurate, and optimized for edge devices.

## 2. Logistic Regression – Irrigation Scheduling

- Type: Binary Classification Model
- Features: Temp, Humidity, Rainfall, Soil Moisture, etc.
- Output: Binary (Irrigation Required / Not Required)
- Accuracy: ~85%
- Reason: Interpretable, efficient, and suitable for real-time prediction.

## 3. Additional Models (for comparison)

- Naive Bayes: Simple and fast but less accurate
- Support Vector Machine (SVM): Good for high-dimensional data but slower
- Random Forest: Tried for feature importance but was heavier than logistic regression
- Resnet50 : Achieved highest accuracy of 95% but was heavyweight
- VGG16: Achieved lowest accuracy.

# 📊 Datasets Used

1. Plant Village Dataset: Contains labeled images of healthy and diseased plants. 
2. Irrigation Scheduling Dataset: Tabular data including temperature, humidity, rainfall, and soil moisture for irrigation prediction.
(All datasets sourced from Kaggle)

# 🧰 Tech Stack

- Frontend: React, Vite, TypeScript
- Backend: Flask API (Python)
- ML/DL: TensorFlow, Keras, Scikit-learn
- Deployment: Localhost / Future scope for cloud deployment

# 🚀 Features

- Real-time plant disease classification with high accuracy
- Smart irrigation recommendation system
- User-friendly interface
- Secure and scalable backend

# 📷 Project Demo
![WhatsApp Image 2025-03-20 at 22 18 02_3f12eb2e](https://github.com/user-attachments/assets/2922b39f-d46c-469d-8063-daf6a3c5ac9c)
![WhatsApp Image 2025-03-20 at 22 19 19_55a12dbe](https://github.com/user-attachments/assets/763c4a56-5f9a-47ac-a0ca-a59ba8cc43e6)
![WhatsApp Image 2025-03-20 at 22 20 04_6f847494](https://github.com/user-attachments/assets/08b64672-4551-4c1b-8171-909cfb9bcc9d)

# 📌 Future Scope

- Integration with IoT sensors for real-time soil and weather monitoring
- Deployment on mobile devices
- Enhanced dashboard with trend analytics

# 🤝 Contributors
DEV PATEL [@devpatel0005](https://github.com/devpatel0005)

HARI PATEL [@haripatel07](https://github.com/haripatel07)

HET PATEL [@ihetpatel](https://github.com/ihetpatel)

