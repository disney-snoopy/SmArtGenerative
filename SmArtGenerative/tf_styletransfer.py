import os
import tensorflow as tf
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import PIL.Image
import time
from tqdm import tqdm

from SmArtGenerative.image_utils import *


def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content':content_dict, 'style':style_dict}


class Transfer():

  def __init__(self, content_img, style_img, **kwargs):
    #Load Content
    self.content_image = content_img
    self.style_image = style_img

    self.kwargs = kwargs

    self.content_layers = self.kwargs.get("content_layers", ['block5_conv2'])

    self.style_layers = self.kwargs.get('style_layers',
                                            ['block1_conv1',
                                            'block2_conv1',
                                            'block3_conv1',
                                            'block4_conv1',
                                            'block5_conv1'])

    self.num_content_layers = len(self.content_layers)
    self.num_style_layers = len(self.style_layers)

    self.extractor = StyleContentModel(self.style_layers, self.content_layers)

    self.style_targets = self.extractor(self.style_image)['style'] #Get Style Layers
    self.content_targets = self.extractor(self.content_image)['content'] #Get Content Layers

    self.var = self.kwargs.get('var', True)
    self.var_weight = self.kwargs.get('var_weight', 30)

    self.style_weight = self.kwargs.get('style_weight', 1e-2)
    self.content_weight = self.kwargs.get('content_weight', 1e4)

    self.n_epochs = self.kwargs.get('n_epochs', 8)
    self.n_steps = self.kwargs.get('n_steps', 100)
    self.store_iter = self.kwargs.get('store_iter', False)
    self.show_image = self.kwargs.get('show_image', False)

    self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


    self.image = tf.Variable(self.content_image)
    self.img_list = []


  def style_content_loss(self, outputs):
    style_outputs = self.outputs['style']
    content_outputs = self.outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) ##MSE
                            for name in style_outputs.keys()])
    style_loss *= self.style_weight / self.num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) ##MSE
                              for name in content_outputs.keys()])
    content_loss *= self.content_weight / self.num_content_layers
    loss = style_loss + content_loss
    return loss

  @tf.function()
  def train_step(self, image):
    with tf.GradientTape() as tape:
      self.outputs = self.extractor(image)
      loss = self.style_content_loss(self.outputs)
      if self.var:
        loss += self.var_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    self.opt.apply_gradients([(grad, image)])
    self.image.assign(clip_0_1(image))


  def transfer(self):
    epochs = self.n_epochs
    steps_per_epoch = self.n_steps

    step = 0
    for n in range(epochs):
      if n == 0:
        print('Extracting style')
      if n == int(epochs/2):
        print('Transfering style')
      for m in range(steps_per_epoch):
        step += 1
        self.train_step(self.image)
        print(".", end='')
      if self.store_iter:
        self.img_list.append(tf.convert_to_tensor(self.image))

      display.clear_output(wait=True)
      if self.show_image:
        display.display(tensor_to_image(self.image))
      print("\nIterations: {}".format(step))

  def reset_image(self):
    '''resets the stored image to content_image'''
    self.image.assign(self.content_image)


  def plot_results(self):
    plt.subplot(1, 3, 1)
    self.imshow(self.content_image, 'Content Image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    self.imshow(self.style_image, 'Style Image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,3,3)
    self.imshow(self.image, 'Result')
    plt.xticks([])
    plt.yticks([])
