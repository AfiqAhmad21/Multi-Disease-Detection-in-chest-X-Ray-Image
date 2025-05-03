import streamlit as st
from PIL import Image
from gradcam_utils import load_model, predict_with_gradcam

st.set_page_config(page_title="Lung Disease Classifier", layout="centered")
st.title("🫁 Multi Lung Disease Classification")
st.markdown("Upload a chest X-ray image to classify and visualize disease regions.")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        model = load_model()
        label, confidence, heatmap = predict_with_gradcam(model, image)

    st.success(f"Predicted Disease: **{label}** ({confidence*100:.2f}%)")
    st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)

