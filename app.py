import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from PIL import Image

classification_model = load_model("skin_disease_cnn_model.h5")
print(classification_model.input_shape)

from tensorflow.keras.applications import VGG16

base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=base_model.input, outputs=base_model.output)
import joblib

kmeans = joblib.load("severity_model.pkl")

class_to_disease = {
    0: 'Actinic keratosis',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Nevus',
    6: 'Vascular lesions'
}

cluster_to_severity = {
    0: 'early',
    1: 'mild',
    2: 'severe'
}

def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_disease_and_severity(image):
   
    img_array = preprocess_image(image)
    
    disease_pred = classification_model.predict(img_array)
    predicted_class = np.argmax(disease_pred, axis=1)
    predicted_disease = class_to_disease[predicted_class[0]]
    
    
    features = feature_model.predict(img_array)
    features_flattened = features.reshape(1, -1)
    
    
    severity = kmeans.predict(features_flattened)
    severity_label = cluster_to_severity[severity[0]]
    
    return predicted_disease, severity_label


st.title("Skin Disease Diagnosis and Severity Prediction")


uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    
    predicted_disease, severity = predict_disease_and_severity(uploaded_image)
    
    
    st.subheader(f"Predicted Disease: {predicted_disease}")
    st.subheader(f"Severity Level: {severity}")



