import os
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from pyflow import pyflow
from PIL import Image
from abc import ABCMeta, abstractmethod


def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7,
                                         nInnerFPIterations=1, nSORIterations=30, colType=0)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2).transpose(2, 0, 1)
    return flow


def aug_patch(central_stack, gt, side_stack, patch_size, scale_factor):
    (input_width, input_height) = central_stack.size
    input_x = random.randrange(0, input_width - patch_size + 1)
    input_y = random.randrange(0, input_height - patch_size + 1)

    gt_patch_size = patch_size * scale_factor
    (gt_x, gt_y) = (input_x * scale_factor, input_y * scale_factor)

    input_box = (input_x, input_y, input_x + patch_size, input_y + patch_size)
    gt_box = (gt_x, gt_y, gt_x + gt_patch_size, gt_y + gt_patch_size)

    central_stack = central_stack.crop(input_box)
    gt = gt.crop(gt_box)
    side_stack = [img.crop(input_box) for img in side_stack]

    return central_stack, gt, side_stack


def aug_flip(central_stack, gt, side_stack):
    if random.random() < 0.5:
        central_stack = central_stack.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        side_stack = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in side_stack]

    if random.random() < 0.5:
        central_stack = central_stack.transpose(Image.FLIP_TOP_BOTTOM)
        gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
        side_stack = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in side_stack]

    return central_stack, gt, side_stack


class BaseDataset(data.Dataset, metaclass=ABCMeta):
    def __init__(self, cfg, is_train=False):
        super(BaseDataset, self).__init__()
        self.sequence_size = cfg.SEQUENCE_SIZE
        self.scale_factor = cfg.SCALE_FACTOR
        self.data_augmentation = cfg.TRAIN.DATASET.DATA_AUGMENTATION if is_train else cfg.TEST.DATASET.DATA_AUGMENTATION
        self.patch_size = cfg.TRAIN.DATASET.PATCH_SIZE if is_train else cfg.TEST.DATASET.PATCH_SIZE
        self.transform = transforms.ToTensor()
        self.is_train = is_train
        self.file_list = None
        self.video_names = None
        self.image_names = None

    def __getitem__(self, idx):
        central_stack, gt, side_stack = self.load_image(self.file_list[idx])

        if self.patch_size != 0:
            central_stack, gt, side_stack = aug_patch(central_stack, gt, side_stack, self.patch_size, self.scale_factor)

        if self.data_augmentation:
            central_stack, gt, side_stack = aug_flip(central_stack, gt, side_stack)

        flow = [get_flow(central_stack, img) for img in side_stack]

        central_stack = self.transform(central_stack)
        gt = self.transform(gt) if gt is not None else gt
        side_stack = [self.transform(img) for img in side_stack]
        flow = [torch.from_numpy(data).float() for data in flow]
        video_name = self.video_names[idx]
        image_name = self.image_names[idx]

        output_dict = {
            'central_stack': central_stack,
            'side_stack': side_stack,
            'flow': flow,
            'video_name': video_name,
            'image_name': image_name,
        }
        if gt is not None:
            output_dict['gt'] = gt

        return output_dict

    def __len__(self):
        return len(self.file_list)

    @abstractmethod
    def load_image(self, data_dir):
        pass
