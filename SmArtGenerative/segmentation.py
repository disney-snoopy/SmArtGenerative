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

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'


class Segmentation():
    def __init__(self, model_path):
        # pretrained model path is required
        self.model_path = model_path
        self.model = torch.load(self.model_path, map_location=map_location).eval()

    def img_transform(self, img):
        '''converts PIL image into torch vector with dummy dimension'''
        tensor = T.ToTensor()(img)
        #add dummy dimension
        tensor = tensor[None, :,:,:]
        return tensor.to(device)

    def run_segmentation(self, content_image):
        '''run segmentation model on input image'''
        # input type is PIL class
        # converting to torch vector with dummy dimension
        self.content_image_PIL = content_image
        self.content_image_vector_dummy = self.img_transform(self.content_image_PIL)
        # run model
        self.pred = self.model(self.content_image_vector_dummy)[0]
        print('Segmentation Complete')

    def crop_obj(self, stylised_image, margin = 30, object_num = 1):
        '''return cropped objects from original content image and stylised image'''
        self.object_num = object_num
        self.stylised_image_PIL = stylised_image
        self.stylised_image_vector_dummy = self.img_transform(self.stylised_image_PIL)
        # assert that both stylised and content images have the same shape
        assert self.content_image_vector_dummy.shape == self.stylised_image_vector_dummy.shape
        ori_shape = self.content_image_vector_dummy.shape

        # currently only crops one object.
        # needs to be updated
        box = self.pred['boxes'][object_num-1]

        a,b,c,d = box

        # adjust box anchor coordinates using
        margin = margin
        self.x_min = np.clip(int(a)-margin, a_min = 0, a_max = ori_shape[-1])
        self.y_min = np.clip(int(b)-margin, a_min = 0, a_max = ori_shape[-2])
        self.x_max = np.clip(int(c)+margin, a_min = 0, a_max = ori_shape[-1])
        self.y_max = np.clip(int(d)+margin, a_min = 0, a_max = ori_shape[-2])

        self.crop_content_tensor = self.content_image_vector_dummy[:,:, self.y_min:self.y_max, self.x_min:self.x_max]
        self.crop_style_tensor = self.stylised_image_vector_dummy[:,:, self.y_min:self.y_max, self.x_min:self.x_max]

        return self.crop_content_tensor, self.crop_style_tensor

    def plot_box_ind(self, threshold = 0.7):
        '''Plot segmentation result'''
        boxes = self.pred['boxes'].detach().cpu()
        scores = self.pred['scores'].detach().cpu()
        masks = self.pred['masks']
        plt.figure(figsize=(12,8))
        plt.imshow(self.content_image_PIL)
        target_counter = 0
        # plotting box anchors
        for box, score in zip(boxes, scores):
          if score > threshold:
            plt.scatter(box[0], box[1])
            plt.scatter(box[2], box[3])
            plt.plot([box[0], box[2]], [box[1], box[3]])
            plt.text(box[0], box[1], s = f'{target_counter+1}', fontsize = 12, color = 'black', backgroundcolor = 'gray', alpha = 0.7)
            target_counter += 1
        # subplots for mask contour
        fig, axs = plt.subplots(int(np.ceil(target_counter/4)), 4,
                                figsize = (12, 3*(np.ceil(target_counter/4))),
                                sharex = True, sharey = True)
        axs = axs.flatten()
        ax = 0
        for mask, score in zip(masks, scores):
          if score > threshold:
            axs[ax].imshow(mask.detach().cpu()[0], cmap = 'gray')
            axs[ax].imshow(self.content_image_PIL, alpha = 0.4)
            #axs[ax].set_title('score: %.3f' %score)
            axs[ax].set_xticks([])
            axs[ax].set_yticks([])
            ax += 1
        #fig.suptitle(f'Threshold: {threshold}')
        return fig

    def patch(self, restored_obj_list, mask_threshold = 4e-1):
        self.output_recon = []
        for recon_obj in restored_obj_list:
            # [1, 3, 519, 246]
            obj = recon_obj.detach().cpu().clone()
            # [1, 1280, 956]
            mask = self.pred['masks'][0].detach().cpu().clone()
            # [1, 519, 246]
            mask_cropped = mask[:, self.y_min:self.y_max, self.x_min:self.x_max]
            # [1, 519, 246]
            binary_mask = (mask_cropped > mask_threshold)[0]

            # [1, 3, 1280, 959]
            stylised = self.stylised_image_vector_dummy.clone().cpu()
            # [1, 3, 519, 246]
            stylised_crop = stylised[:,:, self.y_min:self.y_max, self.x_min:self.x_max]

            mask_object = obj * binary_mask
            mask_surrounding = stylised_crop * (binary_mask == False)
            merged = mask_object + mask_surrounding

            outcome_img = self.stylised_image_vector_dummy.clone()
            outcome_img[:,:, self.y_min:self.y_max, self.x_min:self.x_max] = merged

            self.output_recon.append(outcome_img)
        print('Patching complete')