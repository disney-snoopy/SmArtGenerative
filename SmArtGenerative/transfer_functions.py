import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
import PIL.Image
from SmArtGenerative.image_utils import *


def multiple_styles(content_path, style_path_list):
  stylenames = []
  results = []
  for index, style in enumerate(style_path_list):
    stylename = os.path.split(style)[1]
    stylenames.append(stylename)
    print(f"Transfering style from {stylename}")
    model = Transfer(content_path, style, n_epochs=2,show_image=False)
    model.transfer()
    results.append(tf.convert_to_tensor(model.image))
    # del model
    # tf.keras.backend.clear_session()

  print("Content Image:")
  display.display(tensor_to_image(load_img(content_path)))

  # for ind,image in enumerate(results):
  #   print(f"Style: {stylenames[ind]}")

  #   display.display(tensor_to_image(image))
  ncols=2
  nrows= int(np.ceil(len(results)/ncols))

  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

  axs = axes.flatten()
  for index,img in enumerate(results):
    img = tf.squeeze(img, axis=0)
    axs[index].imshow(img)
    axs[index].set_xticks([])
    axs[index].set_yticks([])
    axs[index].set_title(f'Style: {stylenames[index]}')

  # fig.tight_layout()
  plt.show()


def param_search(content_path, style_path, n_epochs, n_comb=5, start_style_weight=1e-2, start_content_weight=1e-2,
                  stop_style_weight= 1e6, stop_content_weight=1e4, var=True, var_weight=30):

  style_weights = np.geomspace(start_style_weight, stop_style_weight, num=10)
  content_weights = np.geomspace(start_content_weight, stop_content_weight, num=10)

  style_weight_list = []
  content_weight_list = []

  img_list =[]

  for i in range(n_comb):
    style_weight = np.random.choice(style_weights)
    content_weight = np.random.choice(content_weights)

    style_weight_list.append(style_weight)
    content_weight_list.append(content_weight)

    model = Transfer(content_path, style_path, style_weight=style_weight,
      content_weight=content_weight, n_epochs=n_epochs, store_iter=True, show_image=False)
    model.transfer()
    img_list += model.img_list
    # reset_image()

  cols = ['Epoch {:}'.format(col) for col in range(1, n_epochs+1)]
  rows = ['Style Weight {:.3f}\nContent Weight {:.3f}'.format(style, content) for style,content in zip(style_weight_list, content_weight_list)]

  fig, axes = plt.subplots(nrows=len(style_weight_list), ncols=n_epochs, figsize=(15, 15))

  axs = axes.flatten()
  for index,img in enumerate(img_list):
    img = tf.squeeze(img, axis=0)
    axs[index].imshow(img)
    axs[index].set_xticks([])
    axs[index].set_yticks([])

  for ax, col in zip(axes[0], cols):
      ax.set_title(col)

  for ax, row in zip(axes[:,0], rows):
      ax.set_ylabel(row, fontsize='large')

  fig.tight_layout()
  plt.show()
  # for group in img_list:
  #   for img in group:
  #     imshow(img)

def layer_search(content_path, style_path, n_epochs):

  content_layer_list = ['block1_conv2',
                'block2_conv2',
                'block3_conv2',
                'block4_conv2',
                'block5_conv2']

  style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

  img_list =[]

  for i in range(len(content_layer_list)):
    content_layers = [content_layer_list[i]]

    model = Transfer(content_path, style_path, content_layers=content_layers,
                n_epochs=n_epochs, store_iter=False, show_image=False)

    model.transfer()
    img_list.append(tf.convert_to_tensor(model.image))
    # reset_image()


  fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 15))

  axs = axes.flatten()
  for index,img in enumerate(img_list):
    img = tf.squeeze(img, axis=0)
    axs[index].imshow(img)
    axs[index].set_xticks([])
    axs[index].set_yticks([])
    axs[index].set_title(f'Layer {content_layer_list[index]}')


  fig.tight_layout()
  plt.show()
  # for group in img_list:
  #   for img in group:
  #     imshow(img)

