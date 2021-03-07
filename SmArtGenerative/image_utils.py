import os
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from google.cloud import storage
import requests
from io import BytesIO

BUCKET_NAME = 'smartgenerative'

BUCKET_DATA_PATH = 'style/style/'

BASE_URI = 'https://storage.googleapis.com/smartgenerative/'

STYLES = {0:'Alley by the Lake - Leonid Afremov',
              1: 'Young Waterbearer - Adam Styka',
              2: 'Sleep Till Spring - David Michael Hinnebusch',
              3: 'Sunday Afternoon on the Island of La Grande Jatte - Georges Seurat',
              4: 'Beaute Calme et Volupte - Houria Niati',
              5: 'Hands Flowers Eyes',
              6: 'Murnau Street with Women - Wassily Kandinsky',
              7: 'Political Convergence - Jackson Pollock',
              8: 'Mona Lisa - Leonardo da Vinci',
              9: 'The Scream - Edvard Munch',
              10: 'Creation of Adam - Michelangelo',
              11: 'Wheatfield with Cypress Tree - Vincent van Gogh',
              12: 'The Starry Night - Vincent van Gogh',
              13: 'The Great Wave off Kanagawa - Katsushika Hokusai',
              14: 'The Last Supper - Leonardo da Vinci',
              15: 'The Fighting Temeraire - J.M.W. Turner',
              16: 'The Son of Man - Rene Magritte',
              17: 'Nighthawks - Edward Hopper',
              18: 'The New Abnormal - The Stokes',
              19: 'The Garden - Joan Miro',
              20: 'Painting (1933) - Joan Miro'
              }

def tensor_to_image(tensor):
  '''Converts output tensor back to an image'''

  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def load_uploaded_image(image, style=False):
  max_dim = 512
  if style:
    img = image
  else:
    img = PIL.Image.open(image)
  img = tf.convert_to_tensor(img_to_array(img)/255)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def load_styles():
  images = []
  client = storage.Client()
  bucket = client.bucket(BUCKET_NAME)
  blobs = bucket.list_blobs(prefix=BUCKET_DATA_PATH)

  for blob in blobs:
    url = BASE_URI+blob.name
    response = requests.get(url)
    img = PIL.Image.open(BytesIO(response.content))
    images.append(img)
  return images

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)
