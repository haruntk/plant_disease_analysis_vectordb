import tensorflow as tf
from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

app = Flask(__name__)

# 1. ResNet50 Modelini Hazırlama
def load_feature_extractor(output_dim=1024):
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    input_layer = base_model.input
    output_layer = tf.keras.layers.Dense(output_dim, activation="linear")(base_model.output)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

feature_extractor = load_feature_extractor()

# 2. Görüntüden Özellik Çıkartma
def extract_features(image, model):
    img = cv2.resize(image, (224, 224))  # ResNet50 giriş boyutu
    img = preprocess_input(np.expand_dims(img, axis=0))  # Model için uygun hale getirme
    features = model.predict(img)
    return features.flatten().tolist()  # Tek boyutlu bir vektör döndür

@app.route('/extract-features', methods=['POST'])
def extract_features_api():
    file = request.files['image']
    if file:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        features = extract_features(image, feature_extractor)
        return jsonify({'features': features})
    else:
        return jsonify({'error': 'No image provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
