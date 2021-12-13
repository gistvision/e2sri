import os

from .BaseDataset import BaseDataset

from PIL import Image


class SRDataset(BaseDataset):
    def __init__(self, root, cfg, is_train=True):
        super(SRDataset, self).__init__(cfg, is_train)
        root = os.path.abspath(root)
        data_root = os.path.join(root, 'train' if is_train else 'test')
        self.file_list = []
        self.video_names = []
        self.image_names = []
        for video_name in os.listdir(data_root):
            for image_name in os.listdir(os.path.join(data_root, video_name)):
                self.file_list.append(os.path.join(data_root, video_name, image_name))
                self.video_names.append(video_name)
                self.image_names.append(image_name)
        self.gt_name = 'target256.png' if self.scale_factor == 2 else 'target512.png'

    def load_image(self, data_dir):
        central_stack = Image.open(os.path.join(data_dir, 'center.png')).convert('RGB')
        gt = Image.open(os.path.join(data_dir, self.gt_name)).convert('RGB')
        interval = self.sequence_size // 2
        side_stack = [Image.open(os.path.join(data_dir, 'center%+d.png' % (idx - interval))).convert('RGB') for idx in
                      range(self.sequence_size) if idx != interval]

        return central_stack, gt, side_stack
