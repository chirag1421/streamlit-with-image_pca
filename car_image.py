import streamlit as st
from PIL import Image, ImageChops
import os
import io,base64

from io import BytesIO
@st.cache
def load_image(photo):
    image = Image.open(photo)

    return image

st.header("car image uploader")

select_box = st.selectbox()


photo = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"])

if photo is not None:
    # To See details
    st.write(type(photo))
    # Methods/Attrib
    # st.write(dir(image_file))
    file_details = {"filename": photo.name, "filetype": photo.type, "filesize": photo.size}
    st.write(file_details)
    st.image(load_image(photo), width=250)
    with open(os.path.join("/Users/chirag/PycharmProjects/carimage_pca/GrayScale_Resized", photo.name), "wb") as f:
        f.write(photo.getbuffer())

    print(photo)
    st.success("Saved File")




