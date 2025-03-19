import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ---- Load Models ----
alexnet_model = tf.keras.models.load_model('AlexNet2.1.keras')
inception_model = tf.keras.models.load_model('multiclassinceptionmodelnew.keras')

# ---- Preprocessing ----
def preprocess(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---- Predict Function ----
def predict(image):
    img = preprocess(image)

    # Get predictions
    output1 = alexnet_model.predict(img)
    output2 = inception_model.predict(img)

    pred1 = np.argmax(output1, axis=1)[0]
    pred2 = np.argmax(output2, axis=1)[0]

    # Max voting
    final_pred = np.bincount([pred1, pred2]).argmax()

    classes = ['BCC', 'BKL', 'MEL']
    return classes[final_pred]

# ---- Streamlit Frontend ----
st.title("Real-Time Skin Lesion Classification")

uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        label = predict(image)
        st.success(f"Prediction: {label}")

