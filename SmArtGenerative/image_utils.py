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


STYLES = {0: {
                'name': 'Alley by the Lake - Leonid Afremov',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              1: {
                'name': 'Young Waterbearer - Adam Styka',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              2: {
                'name': 'Sleep Till Spring - David Michael Hinnebusch',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              3: {
                'name': 'Sunday Afternoon on the Island of La Grande Jatte - Georges Seurat',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              4: {
                'name': 'Beaute Calme et Volupte - Houria Niati',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              5: {
                'name': 'Hands Flowers Eyes',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              6: {
                'name': 'Murnau Street with Women - Wassily Kandinsky',
                'style': [1e4, 1e1],
                'balanced': [1e3, 1e8],
                'content': [1e1, 1e8]
              },
              7: {
                'name': 'Political Convergence - Jackson Pollock',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              8: {
                'name': 'Mona Lisa - Leonardo da Vinci',
                'style': [1e4, 1e1],
                'balanced': [1e3, 1e8],
                'content': [1e1, 1e8]
              },
              9: {
                'name': 'The Scream - Edvard Munch',
                'style': [1e3, 1e8],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              10: {
                'name': 'Creation of Adam - Michelangelo',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              11: {
                'name': 'Wheatfield with Cypress Tree - Vincent van Gogh',
                'style': [1e4, 1e1],
                'balanced': [1e3, 1e8],
                'content': [1e1, 1e8]
              },
              12: {
                'name': 'The Starry Night - Vincent van Gogh',
                'style': [1e4, 1e1],
                'balanced': [1e3, 1e8],
                'content': [1e1, 1e8]
              },
              13: {
                'name': 'The Great Wave off Kanagawa - Katsushika Hokusai',
                'style': [1e4, 1e1],
                'balanced': [1e3, 1e8],
                'content': [1e1, 1e8]
              },
              14: {
                'name': 'The Last Supper - Leonardo da Vinci',
                'style': [1e3, 1e8],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              15: {
                'name': 'The Fighting Temeraire - J.M.W. Turner',
                'style': [1e2, 1e8],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              16: {
                'name': 'The Son of Man - Rene Magritte',
                'style': [1e3, 1e8],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
              },
              17: {
                'name': 'Nighthawks - Edward Hopper',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
                },
              18: {
                'name': 'The New Abnormal - The Strokes',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
                },
              19: {
                'name': 'The Garden - Joan Miro',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
                },
              20: {
                'name': 'Painting (1933) - Joan Miro',
                'style': [1e4, 1e1],
                'balanced': [1e1, 1e8],
                'content': [1e-1, 1e8]
                }
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
  max_dim = 750
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
  max_dim = 750
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

# def get_credentials():
#     credentials_raw = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
#     if '.json' in credentials_raw:
#         credentials_raw = open(credentials_raw).read()
#     creds_json = json.loads(credentials_raw)
#     creds_gcp = service_account.Credentials.from_service_account_info(creds_json)
#     return creds_gcp

def load_styles():
  #credentials = get_credentials()
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

def load_styles_local():
  # return array of images
  path = os.path.join(os.path.dirname(__file__),'data/style')
  imagesList = sorted(os.listdir(path))
  loadedImages = []
  for image in imagesList:
    img = PIL.Image.open(path + '/' + image)
    loadedImages.append(img)

  return loadedImages

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
