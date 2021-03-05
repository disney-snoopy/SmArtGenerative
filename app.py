import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from SmArtGenerative.utils import loader, unloader
from SmArtGenerative.trainer_segmentation import TrainerSegmentation
from SmArtGenerative.params import vgg_model_path, segmentation_model_path
from SmArtGenerative.image_utils import load_uploaded_image, load_img, tensor_to_image
from SmArtGenerative.transfer_functions import multiple_styles
from SmArtGenerative.tf_styletransfer import Transfer
import streamlit.components.v1 as components

# streamlit setting
st.set_option('deprecation.showfileUploaderEncoding', False)

# page title
st.title("SmArt Generative Service")
st.write("")

# dummy variables for if statement
forward_final = None
fig = None


##### OMER UPLOAD OPTIONS #########

content_up = st.file_uploader("Upload an image:", type=['png', 'jpg', 'jpeg'])

if content_up is not None:

    content_img = load_uploaded_image(content_up)

    options = ['I want to upload my own style image',
                'Choose a style from the gallery',
                'Surprise me!']

    option = st.selectbox('Pick a style option', options)
    if option == 'I want to upload my own style image':
        style_up = st.file_uploader("Upload the image to transfer style from", type=['png', 'jpg', 'jpeg'])

        if style_up is not None:
            style_img = load_uploaded_image(style_up)

            weights = st.radio('Preference', ('Style Heavy', 'Balanced', 'Content Heavy'))

            model = Transfer(content_img, style_img, n_epochs=1)
            model.transfer()
            img = tensor_to_image(model.image)
            st.image(img, 'Voila')

    if option == 'Choose a style from the gallery':

        pics = [x for x in range(1,21)]
        st.write('You can pick from the following styles:')
        st.image('https://storage.googleapis.com/smartgenerative/style/style-gallery.png')
        st.image('https://storage.googleapis.com/smartgenerative/style/style-gallery-2.png')
        option = st.selectbox('Which style would you like', pics)

    if option == 'Surprise me!':
        pass

##### OMER UPLOAD OPTIONS #########

