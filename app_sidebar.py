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

check_segmentation = True
if st.sidebar.checkbox('Restore people resolution', value = True):
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
crop_boolean = None
num_objects = None
run_restoration = None




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

# Display style image once uploaded
if style_up is not None:
    image_style = Image.open(style_up)
    tensor_style = loader(style_up)
    c2.image(image_style, caption='Your Style Image.', use_column_width=True)

# Once both content and style pictures are uploaded and segmentation check box is true,
# run segmentation and display potential choices.

@st.cache
def get_segmentation_result(tensor_content_resized, tensor_style, vgg_model_path, segmentation_model_path):
    trainer = TrainerSegmentation(tensor_content=tensor_content_resized,
                                  tensor_style=tensor_style,
                                  path_vgg=vgg_model_path,
                                  path_seg=segmentation_model_path)
    trainer.segmentation()
    return trainer

if content_up is not None and style_up is not None and check_segmentation == True:
    trainer = get_segmentation_result(tensor_content_resized, tensor_style, vgg_model_path, segmentation_model_path)
    fig, num_objects = trainer.seg.plot_box_ind(threshold = 0.4)
    c3.pyplot(fig, caption = 'Human object in your picture!', use_column_width = True)


###############################
# Need to interactively fetch object indices which users want to keep
# Number of choices given to the user can vary

if num_objects is not None:
    c3.write(f'We found {num_objects} possible human objects!\n Choose the ones you want to restore in the dropdown menu!')
    object_idx = st.sidebar.multiselect('Choose the objects numbers you want to maintain', range(num_objects))

if st.sidebar.button('Run Detail Restoration'):
    # print is visible in server output, not in the page
    run_restoration = 1
    c3.write(f'Restoration is in progress!')

###############################
# once run restoration button is pushed, run restoration

if run_restoration is not None:
    #if check_gif is True:
     #   trainer.stylise(style_weight = (10 ** style_weight), epochs = num_epoch, output_freq = int(num_epochs/20))
    #else:
    trainer.stylise(style_weight = (10 ** style_weight), epochs = num_epoch, output_freq = num_epoch)

    forward_final = trainer.forward_final

    c3.image(forward_final, caption = 'Style Transfer Complete', use_column_width=True)

if forward_final is not None and check_segmentation == True:

    trainer.seg_crop(object_idx = object_idx)
    trainer.content_reconstruction(lr = 0.0005, epochs = restoration_epoch)
    trainer.patch()
    reverse_final = trainer.reverse_final
    c3.image(reverse_final, caption = 'Your contour restored', use_column_width=True)