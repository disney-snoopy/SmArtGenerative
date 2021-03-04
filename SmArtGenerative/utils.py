from PIL import Image

import torch
import torchvision.transforms as T


def loader(path):
  '''Convert PIL image into torch tensor with dummy dimension'''
  preproc = T.Compose([T.ToTensor()])
  img = Image.open(path)
  img_proc = preproc(img).unsqueeze(0)
  return img_proc

def unloader(tensor_img):
  '''convert torch tensor into PIL image'''
  t_unload = T.ToPILImage()
  img = t_unload(tensor_img.squeeze())
  return img

def gram_matrix(input_tensor):
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    # calculate gram matrix
    G = torch.mm(features, features.t())
    # normalise
    return G.div(a * b * c * d)