import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import cv2
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from datetime import datetime

# ResNet50 Modelini Hazırlayın
def load_feature_extractor(output_dim=1024):
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    input_layer = base_model.input
    output_layer = tf.keras.layers.Dense(output_dim, activation="linear")(base_model.output)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

feature_extractor = load_feature_extractor()

# Görüntüden Özellik Çıkartma
def extract_features(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    img = cv2.resize(img, (224, 224))  # ResNet50 giriş boyutu
    img = preprocess_input(np.expand_dims(img, axis=0))  # Model için uygun hale getirme
    features = model.predict(img)
    return features.flatten()  # Tek boyutlu vektör döndür

# Database ve Koleksiyon Oluşturma
def create_or_get_collection(db_name, collection_name, vector_dim):
    connections.connect(alias="default", host="127.0.0.1", port="19530", db_name=db_name)
    
    # Koleksiyon mevcut mu?
    if not utility.has_collection(collection_name):
        print(f"{collection_name} koleksiyonu oluşturuluyor...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="camera_data_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=255)
        ]
        schema = CollectionSchema(fields, description="Görüntü vektör koleksiyonu")
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)
    
    return collection

# Görüntüleri Yükleme
def load_images_to_collection(dataset_path, collection, model):
    camera_data_id = 2
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            try:
                # Görüntüden özellik çıkar
                feature_vector = extract_features(img_path, model)
                feature_vector_list = feature_vector.tolist()

                # Görüntü dosyasının tarihi alınır
                timestamp = os.path.getmtime(img_path)
                creation_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

                # Koleksiyona ekle
                collection.insert([
                    [feature_vector_list],
                    [camera_data_id],  # camera_data_id olarak numaralandırma
                    [category],  # category olarak klasör adı
                    [creation_date]  # date olarak oluşturulma tarihi
                ])
                print(f"{img_file} -> {category} kategorisi altında eklendi.")
                camera_data_id += 1
            except Exception as e:
                print(f"Hata ({img_file}): {e}")

# Ana Akış
if __name__ == "__main__":
    dataset_path = "C:/Users/harun_rvth/OneDrive/Desktop/KameraData"  # Görüntü klasörü
    db_name = "AIGreenhouse"  # Veritabanı adı (Milvus'ta sadece bağlantıda kullanılır)
    collection_name = "Leaves"  # Koleksiyon adı
    vector_dim = 1024  # Vektör boyutu

    # Koleksiyon oluştur veya getir
    collection = create_or_get_collection(db_name, collection_name, vector_dim)
    
    # Görüntüleri koleksiyona yükle
    load_images_to_collection(dataset_path, collection, feature_extractor)

    print("Tüm görüntüler başarıyla yüklendi!")
