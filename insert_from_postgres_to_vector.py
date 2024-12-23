import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import cv2
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import psycopg2
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
 
# PostgreSQL Bağlantısı ve Verileri Alma
def fetch_camera_data():
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="db_name",
            user="postgres",
            password="pswrd"
        )
        cursor = connection.cursor()
        cursor.execute("SELECT id, file_name, image_date, camera_id FROM public.camera_data;")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"PostgreSQL bağlantı hatası: {e}")
        rows = []
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()
    return rows
 
# Database ve Koleksiyon Oluşturma
def create_or_get_collection(db_name, collection_name, vector_dim):
    connections.connect(alias="default", host="127.0.0.1", port="19530", db_name=db_name)
    # Koleksiyon mevcut mu?
    if not utility.has_collection(collection_name):
        print(f"{collection_name} koleksiyonu oluşturuluyor...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="camera_data_id", dtype=DataType.INT64),  # Veritabanındaki id alanı
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=255)
        ]
        schema = CollectionSchema(fields, description="Görüntü vektör koleksiyonu")
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)
    return collection
 
# Kategoriyi Dosya Adından Çekme
def extract_category_from_filename(file_name):
    category = file_name.split('/')[0]  # Dosya adındaki ilk kısmı al (FV-8 gibi)
    return category
 
# Görüntüleri Yükleme
def load_images_to_collection(camera_data, dataset_path, collection, model):
    for data in camera_data:
        id_, file_name, image_date, camera_id = data
        img_path = os.path.join(dataset_path, file_name)
        try:
            # Dosya adından kategori al
            category = extract_category_from_filename(file_name)
 
            # Tarihi string formatına çevir
            image_date_str = image_date.strftime('%Y-%m-%d') if isinstance(image_date, datetime) else str(image_date)
 
            # Görüntüden özellik çıkar
            feature_vector = extract_features(img_path, model)
            feature_vector_list = feature_vector.tolist()
 
            # Koleksiyona ekle
            collection.insert([
                [feature_vector_list],
                [id_],  # camera_data_id veritabanındaki id ile eşleştirildi
                [category],  # category (dosya adı ile belirlenen kategori)
                [image_date_str]  # date
            ])
            print(f"{file_name} -> Koleksiyona başarıyla eklendi.")
        except Exception as e:
            print(f"Hata ({file_name}): {e}")
 
# Ana Akış
if __name__ == "__main__":
    dataset_path = "path"  # Görüntü klasörü
    db_name = "AIGreenhouse"  # Veritabanı adı (Milvus'ta sadece bağlantıda kullanılır)
    collection_name = "Leaves"  # Koleksiyon adı
    vector_dim = 1024  # Vektör boyutu
 
    # PostgreSQL'den verileri al
    camera_data = fetch_camera_data()
 
    # Koleksiyon oluştur veya getir
    collection = create_or_get_collection(db_name, collection_name, vector_dim)
    # Görüntüleri koleksiyona yükle
    load_images_to_collection(camera_data, dataset_path, collection, feature_extractor)
 
    print("Tüm görüntüler başarıyla yüklendi!")