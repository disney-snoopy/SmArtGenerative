import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from SmArtGenerative.utils import loader, unloader
from SmArtGenerative.trainer_segmentation import TrainerSegmentation
from SmArtGenerative.params import vgg_model_path, segmentation_model_path


# setting wide screen
st.set_page_config(layout="wide")

# title
st.title("SmArt Generative Service")

# multi column layout
c1, c2, c3 = st.beta_columns((1, 1, 3))
c1.header("Content picture")
c2.header("Style picture")

# sidebar for interactions
style_weight = st.sidebar.slider('Style Weight', 5, 20, 10)
num_epoch = st.sidebar.slider(label = 'Number of iterations',
                              min_value = 100,
                              max_value = 600,
                              value = 300,
                              step = 50)

check_segmentation = False
if st.sidebar.checkbox('Restore people resolution'):
    check_segmentation = True

restoration_epoch = st.sidebar.slider(label = 'Restoration strength',
                                      min_value = 100,
                                      max_value = 600,
                                      value = 300,
                                      step = 50)

check_gif = False
if st.sidebar.checkbox('Export transformation gif'):
    check_gif = True

#dummy variables for if statements
forward_final = None
fig = None




# img upload
content_up = c1.file_uploader("Upload your picture for style transfer", type=['png', 'jpg', 'jpeg'])
style_up = c2.file_uploader("Upload your favourite style picture", type=['png', 'jpg', 'jpeg'])

# Once content picture is uploaded, resize and display
if content_up is not None:
    image_content = Image.open(content_up)
    tensor_content_resized = T.ToTensor()(image_content).unsqueeze(0)
    tensor_content_resized = T.functional.resize(tensor_content_resized,
                                                [int(tensor_content_resized.shape[-2]/3), int(tensor_content_resized.shape[-1]/3)])
    image_resized = unloader(tensor_content_resized)
    c1.image(image_resized, caption='Image to Stylise.', use_column_width=True)

# SDisplay style image once uploaded
if style_up is not None:
    image_style = Image.open(style_up)
    tensor_style = loader(style_up)
    c2.image(image_style, caption='Your Style Image.', use_column_width=True)

if content_up is not None and style_up is not None:
    c3.write('Style transfer in progress!')
    trainer = TrainerSegmentation(tensor_content=tensor_content_resized,
                                  tensor_style=tensor_style,
                                  path_vgg=vgg_model_path,
                                  path_seg=segmentation_model_path)
    if check_gif is True:
        trainer.stylise(style_weight = (10 ** style_weight), epochs = num_epoch, output_freq = int(num_epochs/20))
    else:
        trainer.stylise(style_weight = (10 ** style_weight), epochs = num_epoch, output_freq = num_epoch)

    forward_final = trainer.forward_final

    if check_segmentation is False:
        c3.image(forward_final, caption = 'Style Transfer Complete', use_column_width=True)

if forward_final is not None and check_segmentation == True:
    trainer.segmentation()
    fig = trainer.seg.plot_box_ind()
    c3.pyplot(fig, caption = 'Human object in your picture!', use_column_width = True)

if fig is not None:
    trainer.content_reconstruction(lr = 0.0005, epochs = restoration_epoch)
    trainer.patch()
    reverse_final = trainer.reverse_final
    c3.image(reverse_final, caption = 'Your contour restored', use_column_width=True)