import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import cv2
from pymilvus import connections, Collection
from pymilvus import utility

# 1. ResNet50 Modelini Hazırlama
def load_feature_extractor(output_dim=1024):
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    input_layer = base_model.input
    output_layer = tf.keras.layers.Dense(output_dim, activation="linear")(base_model.output)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

feature_extractor = load_feature_extractor()

# 2. Görüntüden 128 Boyutlu Özellik Çıkartma
def extract_features(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # ResNet50 giriş boyutu
    img = preprocess_input(np.expand_dims(img, axis=0))  # Model için uygun hale getirme
    features = model.predict(img)
    return features.flatten()  # Tek boyutlu bir vektör döndür

# 3. Mevcut Milvus Koleksiyonuna Bağlanma ve Vektör Ekleme
def add_vector_to_existing_collection(db_name,collection_name, feature_vector, category):
    # Milvus'a bağlan
    connections.connect("default", host="127.0.0.1", port="19530",db_name=db_name)
    
    # Mevcut koleksiyonu kontrol et
    if not utility.has_collection(collection_name):
        print(f"Hata: {collection_name} koleksiyonu mevcut değil.")
        return

    # Koleksiyonu seç
    collection = Collection(name=collection_name)

    # Vektörü koleksiyona ekle
    insert_result = collection.insert([[feature_vector_list], [category]])
    print(f"Vektör başarıyla {collection_name} koleksiyonuna eklendi. ID: {insert_result.primary_keys}")


# Ana Akış
if __name__ == "__main__":
    # Görüntü yolu
    image_path = "some/path"  # Analiz edilecek görüntü dosyası

    # Özellik çıkartma
    feature_vector = extract_features(image_path, feature_extractor)
    feature_vector_list = feature_vector.tolist()
    # Mevcut Milvus koleksiyonuna ekle
    db_name = "AIGreenhouse"  # Veritabanı adı
    collection_name = "Leaves"  # Koleksiyon adı
    category = "healty"
    add_vector_to_existing_collection(db_name, collection_name, feature_vector_list ,category)

    print(f"1024 boyutlu vektör, {db_name} veritabanındaki {collection_name} koleksiyonuna başarıyla eklendi!")
