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

class LBFGS_Transfer():
    def __init__(self, model_path = None, content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        self.model_path = model_path
        self.content_layers = content_layers
        self.style_layers = style_layers


    def style_loader(self, image_name):
        # fake batch dimension required to fit network's input dimensions
        image = style_transform(image_name).unsqueeze(0)
        return image.to(device, torch.float)

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img):

        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses


    def run_style_transfer(self, content_img, style_img, input_img, content_layers, style_layers,
                           normalization_mean=vgg_normalization_mean, normalization_std=vgg_normalization_std,
                           num_steps=300,style_weight=1000000, content_weight=1, output_freq = 50, verbose = 1):
        output_imgs = []
        epoch_nums = []
        if self.model_path == None:
            cnn = models.vgg16(pretrained=True).features.eval().to(device)
        else:
            cnn = torch.load(self.model_path).features.eval().to(device)

        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(cnn,normalization_mean,
                                                                        normalization_std,
                                                                        style_img, content_img)
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    if verbose == 1:
                        print("run {}:".format(run))
                        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                            style_score.item(), content_score.item()))
                if run[0] % output_freq == 0:
                    output_imgs.append(input_img.clone().detach().data.clamp_(0,1))
                    epoch_nums.append(run[0])

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        output_imgs.append(input_img.data.clamp_(0,1))
        epoch_nums.append(run[0])

        return output_imgs, epoch_nums

    def learn(self, content_img, style_img, input_img, num_steps=300, style_weight=1e6, content_weight=1, epochs = 200, output_freq = 20):
        self.img_content = T.ToPILImage()(content_img.squeeze())
        self.img_style = T.ToPILImage()(style_img.squeeze())
        self.style_weight = style_weight
        self.content_weight = content_weight

        content_dim = (content_img.shape[-2], content_img.shape[-1])
        style_transform = T.Compose([T.Resize((content_dim)),
                                     T.ToTensor()])
        style_img = style_transform(self.img_style).unsqueeze(0).to(device, torch.float)

        self.output_imgs, self.epoch_nums = self.run_style_transfer(content_img, style_img, input_img, self.content_layers, self.style_layers,
                                                                    normalization_mean=vgg_normalization_mean, normalization_std=vgg_normalization_std,
                                                                    num_steps=epochs,style_weight=style_weight, content_weight=1,
                                                                    output_freq = output_freq)

    def plot_output(self, img_per_row = 3):
        num_outputs = len(self.output_imgs)
        num_rows = np.ceil(num_outputs/img_per_row)
        fig, axs = plt.subplots(1, 2, figsize = (16, 6), sharey=True, sharex=True)
        axs = axs.flatten()
        axs[0].imshow(self.img_content)
        axs[1].imshow(self.img_style)

        fig, axs = plt.subplots(nrows=int(num_rows), ncols=int(img_per_row), figsize = (16, 8 * img_per_row), sharex=True, sharey=True)
        axs = axs.flatten()
        img_counter = 0

        for ax in axs:
            ax.imshow(T.ToPILImage()(self.output_imgs[img_counter].squeeze()))
            ax.set_title(f'Epoch: {self.epoch_nums[img_counter]+1}', backgroundcolor = 'gray', color = 'white')
            img_counter += 1
            ax.margins(0.05)
            if img_counter == num_outputs:
                break

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        #scientific notation
        sn_style_weight ='{:e}'.format(self.style_weight)
        axs[0].text(30, -120, f'Style weight: {sn_style_weight}\ncontent weight: {self.content_weight}\ncontent layers: {self.content_layers}\nstyle layers: {self.style_layers}', fontsize = 12)

    def search_weight(self, content_img, style_img, input_img, num_steps=100,begin = 5, end = 12):
        self.img_content = T.ToPILImage()(content_img.squeeze())
        self.img_style = T.ToPILImage()(style_img.squeeze())

        content_dim = (content_img.shape[-2], content_img.shape[-1])
        style_transform = T.Compose([T.Resize((content_dim)),
                                     T.ToTensor()])

        style_img = self.img_style.unsqueeze(0).to(device, torch.float)

        nums = end - begin + 1
        self.style_weights = np.logspace(begin, end, num=nums)
        self.output_imgs = []
        for style_weight in self.style_weights:
            inp_img = input_img.clone()
            output_img, epoch_nums = self.run_style_transfer(content_img, style_img, inp_img, self.content_layers, self.style_layers,
                                                            normalization_mean=vgg_normalization_mean, normalization_std=vgg_normalization_std,
                                                            num_steps=num_steps,style_weight=style_weight, content_weight=1, output_freq = num_steps, verbose = 0,
                                                            model_path = self.model_path)
            self.output_imgs.append(output_img)
        #Plotting input images
        fig, axs = plt.subplots(1, 2, figsize = (16, 6), sharey=True, sharex=True)
        axs = axs.flatten()
        axs[0].imshow(self.img_content)
        axs[1].imshow(self.img_style)

        #Plotting output images
        img_per_row = 3
        num_rows = int(np.ceil(len(self.output_imgs)/img_per_row))

        fig, axs = plt.subplots(nrows=num_rows, ncols=int(img_per_row), figsize = (6*img_per_row, 6*num_rows), sharex=True, sharey=True)
        fig.suptitle(f'Epochs: {num_steps}', fontsize=12)

        axs = axs.flatten()
        img_counter = 0
        for idx, img in enumerate(self.output_imgs):
            axs[idx].imshow(T.ToPILImage()(img[-1].squeeze()))
            axs[idx].set_title(f'style_weight: {self.style_weights[idx]}', backgroundcolor = 'gray', color = 'white')
        axs[0].text(30, -120, f'content weight: {1}\ncontent layers: {self.content_layers}\nstyle layers: {self.style_layers}', fontsize = 12)