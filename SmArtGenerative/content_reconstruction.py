from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models

from SmArtGenerative.utils import *
from SmArtGenerative.params import *
from SmArtGenerative.layers import *

class Content_Reconstructor(nn.Module):
    #Extract__nn class returns the feature maps of the first 5 conv layers of vgg16.
    def __init__(self):
        super(Content_Reconstructor, self).__init__()
        self.conv1 = layers[0]
        self.conv2 = layers[2]
        self.conv3 = layers[5]
        self.conv4 = layers[7]
        self.conv5 = layers[10]
        self.maxpool = layers[4]

    def forward(self, x):
        '''Input is torch tensor with dummy dimension'''
        self.content_image_tensor_dummy = x
        out1 = self.conv1(x)
        out1 = F.relu(out1)
        out2 = self.conv2(out1)
        out2 = F.relu(out2)
        out3 = self.maxpool(out2)
        out3 = self.conv3(out3)
        out3 = F.relu(out3)
        out4 = self.conv4(out3)
        out4 = F.relu(out4)
        out5 = self.maxpool(out4)
        out5 = self.conv5(out5)
        out5 = F.relu(out5)
        self.f_map = out4.detach()

    def model_construct(self, layer_count=9):
        '''Construct minimal model for content reconstruction'''
        vgg16 = models.vgg16(pretrained=True).features.eval()
        layers = list(vgg16.children())
        del vgg16
        model = nn.Sequential()
        counter = 0
        for i in range(layer_count):
            layer_name = f'layer_{counter}'
            counter += 1
            model.add_module(layer_name, layers[i])
        return model

    def restore(self, tensor_stylised, epochs, output_freq, lr = 0.0002, verbose=0):
        '''return content-reconstructed image'''
        #Creating whitenoise image as a starting template
        self.tensor_stylised = tensor_stylised.detach()
        img_start = self.tensor_stylised.clone().requires_grad_()

        #Using MSE loss
        criterion = nn.MSELoss()
        #Using Adam as an optimiser
        opt = optim.Adam(params= [img_start], lr = lr)
        #Instantiating model with given layers from pretrained VGG16
        model = self.model_construct().to(device)

        self.output_imgs = []
        for epoch in range(epochs):
            pred = model(img_start)
            loss = criterion(pred, self.f_map)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if epoch % output_freq == 0:
                self.output_imgs.append(img_start.detach().cpu().data.clamp_(0,1))
            if verbose == 1:
                if epoch % 20 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}')
        self.output_imgs.append(img_start.detach().cpu().data.clamp_(0,1))