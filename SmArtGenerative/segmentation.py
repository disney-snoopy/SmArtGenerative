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


if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

class Segmentation():
    def __init__(self, model_path = None):
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

    def crop_obj(self, stylised_image, margin = 30, object_idx = [0]):
        '''return cropped objects from original content image and stylised image'''
        self.object_idx = object_idx
        self.stylised_image_PIL = stylised_image
        self.stylised_image_vector_dummy = self.img_transform(self.stylised_image_PIL)
        # assert that both stylised and content images have the same shape
        assert self.content_image_vector_dummy.shape == self.stylised_image_vector_dummy.shape
        ori_shape = self.content_image_vector_dummy.shape

        # currently only crops one object.
        # needs to be updated
        self.crop_content_tensor_list = []
        self.crop_style_tensor_list = []
        self.crop_coordinates = []
        for idx in object_idx:
          box = self.pred['boxes'][idx]
          a,b,c,d = box

          # adjust box anchor coordinates using
          margin = margin
          x_min = np.clip(int(a)-margin, a_min = 0, a_max = ori_shape[-1])
          y_min = np.clip(int(b)-margin, a_min = 0, a_max = ori_shape[-2])
          x_max = np.clip(int(c)+margin, a_min = 0, a_max = ori_shape[-1])
          y_max = np.clip(int(d)+margin, a_min = 0, a_max = ori_shape[-2])
          self.crop_coordinates.append((x_min, y_min, x_max, y_max))

          self.crop_content_tensor_list.append(self.content_image_vector_dummy[:,:, y_min:y_max, x_min:x_max])
          self.crop_style_tensor_list.append(self.stylised_image_vector_dummy[:,:, y_min:y_max, x_min:x_max])

        return self.crop_content_tensor_list, self.crop_style_tensor_list

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
            plt.text(box[0], box[1], s = f'{target_counter}', fontsize = 12, color = 'black', backgroundcolor = 'gray', alpha = 0.7)
            target_counter += 1
        # subplots for mask contour
        fig, axs = plt.subplots(int(np.ceil(target_counter/4)), 4,
                                figsize = (12, 2*(np.ceil(target_counter/4))),
                                sharex = True, sharey = True)
        axs = axs.flatten()
        ax = 0
        for mask, score in zip(masks, scores):
          if score > threshold:
            axs[ax].imshow(mask.detach().cpu()[0], cmap = 'gray')
            axs[ax].imshow(self.content_image_PIL, alpha = 0.4)
            axs[ax].set_title('%d - score: %.2f' %(ax, score))
            axs[ax].set_xticks([])
            axs[ax].set_yticks([])
            ax += 1
        #fig.suptitle(f'Threshold: {threshold}')
        return fig, target_counter

    def patch(self, restored_obj_history_list, mask_threshold = 4e-1):
        self.output_recon = []
        outcome_img = self.stylised_image_vector_dummy.clone()
        for num_epoch in range(len(restored_obj_history_list[0])):
            for history_idx, obj_idx in zip(range(len(restored_obj_history_list)), self.object_idx):
                # [1, 3, 519, 246]
                obj = restored_obj_history_list[history_idx][num_epoch].detach().cpu().clone()
                # [1, 1280, 956]
                mask = self.pred['masks'][obj_idx].detach().cpu().clone()
                # [1, 519, 246]
                x_min, y_min, x_max, y_max = self.crop_coordinates[history_idx]
                mask_cropped = mask[:, y_min:y_max, x_min:x_max]

                # [1, 519, 246]
                binary_mask = (mask_cropped > mask_threshold)[0]

                # [1, 3, 1280, 959]
                stylised = self.stylised_image_vector_dummy.clone().cpu()
                # [1, 3, 519, 246]
                stylised_crop = stylised[:,:, y_min:y_max, x_min:x_max]

                mask_object = obj * binary_mask
                mask_surrounding = stylised_crop * (binary_mask == False)
                merged = mask_object + mask_surrounding

                outcome_img[:,:, y_min:y_max, x_min:x_max] = merged
                if obj_idx == max(self.object_idx):
                    self.output_recon.append(outcome_img.clone())
        print('Patching complete')
