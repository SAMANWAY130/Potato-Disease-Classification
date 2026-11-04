import streamlit as st
import numpy as np
import tensorflow as tf


@st.cache_resource
def load_data():
    return tf.keras.models.load_model("potato_disease.keras")

model = load_data()

lab = ["early_blight","late_blight","healthy"]

st.title("Potato Disease Detection")

images = st.file_uploader("Upload potato leaf image", type=["png", "jpg", "jpeg"])
if images is not None :
    img = tf.keras.utils.load_img(images)
    st.image(img,caption="Uploaded image")
    img = tf.keras.utils.img_to_array(img)
    img = tf.image.resize(img, (256, 256))
    img = img / 255
    img = np.expand_dims(img, axis=0)

    if st.button("Predict"):
        pred = model.predict(img)
        ans = np.argmax(pred[0])
        st.success(lab[ans])