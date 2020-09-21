import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from pyflow import pyflow
from PIL import Image


def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7,
                                         nInnerFPIterations=1, nSORIterations=30, colType=0)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2).transpose(2, 0, 1)
    return flow


class SingleFolderDataset(data.Dataset):
    def __init__(self, root, cfg, is_train=False):
        super(SingleFolderDataset, self).__init__()
        self.root = root
        self.sequence_size = cfg.SEQUENCE_SIZE
        self.scale_factor = cfg.SCALE_FACTOR
        self.transform = transforms.ToTensor()
        self.is_train = is_train

        self.file_list = os.listdir(root)
        self.file_list.sort(key=lambda x: int(x[:-4]))

        self.interval = self.sequence_size // 2
        self.file_list = [self.file_list[idx:idx + self.sequence_size]
                          for idx in range(len(self.file_list) - self.sequence_size + 1)]
        self.image_idx = [file[self.interval].split('/')[-1] for file in self.file_list]

    def __getitem__(self, idx):
        central_stack, side_stack = self.load_image(self.file_list[idx])
        flow = [get_flow(central_stack, img) for img in side_stack]

        central_stack = self.transform(central_stack)
        side_stack = [self.transform(img) for img in side_stack]
        flow = [torch.from_numpy(data).float() for data in flow]
        image_idx = self.image_idx[idx]

        return central_stack, side_stack, flow, image_idx

    def __len__(self):
        return len(self.file_list)

    def load_image(self, data_list):
        central_stack = Image.open(os.path.join(self.root, data_list[self.interval])).convert('RGB')
        side_stack = [Image.open(os.path.join(self.root, data_list[idx])).convert('RGB') for idx in
                      range(len(data_list)) if idx != self.interval]

        return central_stack, side_stack
