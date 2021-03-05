import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from SmArtGenerative.utils import loader, unloader
from SmArtGenerative.trainer_segmentation import TrainerSegmentation


# pretrained model paths
vgg_model_path = '/home/byungjae/code/disney-snoopy/SmArtGenerative/pretrained_models/vgg16_pretrained'
seg_model_path = '/home/byungjae/code/disney-snoopy/SmArtGenerative/pretrained_models/torch_segmentation_finetuned'

# streamlit setting
st.set_option('deprecation.showfileUploaderEncoding', False)

# page title
st.title("SmArt Generative Service")
st.write("")

# dummy variables for if statement
forward_final = None
fig = None

# image upload widget
content_up = st.file_uploader("Upload your picture for style transfer", type=['png', 'jpg', 'jpeg'])
style_up = st.file_uploader("Upload your favourite style picture", type=['png', 'jpg', 'jpeg'])

if content_up is not None:
    image_content = Image.open(content_up)
    tensor_content_resized = T.ToTensor()(image_content).unsqueeze(0)
    tensor_content_resized = T.functional.resize(tensor_content_resized,
                                                [int(tensor_content_resized.shape[-2]/2), int(tensor_content_resized.shape[-1]/2)])
    image_resized = unloader(tensor_content_resized)
    st.image(image_resized, caption='Image to Stylise.', use_column_width=True)
    st.write("")

if style_up is not None:
    image_style = Image.open(style_up)
    tensor_style = loader(style_up)
    st.image(image_style, caption='Your Style Image.', use_column_width=True)
    st.write("")

if content_up is not None and style_up is not None:
    st.write('Style transfer in progress!')
    trainer = TrainerSegmentation(tensor_content=tensor_content_resized,
                                  tensor_style=tensor_style,
                                  path_vgg=vgg_model_path,
                                  path_seg=seg_model_path)
    trainer.stylise(style_weight = 1e11, epochs = 50, output_freq = 50)
    forward_final = trainer.forward_final
    st.image(forward_final, caption = 'Style Transfer Complete', use_column_width=True)

if forward_final is not None:
    trainer.segmentation()
    fig = trainer.seg.plot_box_ind()
    st.pyplot(fig, caption = 'Human object in your picture!', use_column_width = True)

if fig is not None:
    trainer.content_reconstruction(lr = 0.0005, epochs = 50)
    trainer.patch()
    reverse_final = trainer.reverse_final
    st.image(reverse_final, caption = 'Your contour restored', use_column_width=True)












#    labels = predict(content_up)
#
 #   # print out the top 5 prediction labels with scores
  #  for i in labels:
   #     st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
