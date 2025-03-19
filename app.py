import streamlit as st

st.title("Skin Lesion Classification App")

st.write("Welcome to the Skin Lesion Classifier! Upload an image to classify.")

# Simple file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    # Dummy prediction output (you can later replace this with your model prediction)
    st.write("Prediction: Melanoma")
