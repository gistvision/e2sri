import os

from .BaseDataset import BaseDataset

from PIL import Image


class SingleFolderDataset(BaseDataset):
    def __init__(self, root, cfg, is_train=True):
        super(SingleFolderDataset, self).__init__(cfg, is_train)
        root = os.path.abspath(root)
        self.file_list = sorted(os.listdir(root))
        self.file_list = [os.path.join(root, file_name) for file_name in self.file_list]
        self.interval = self.sequence_size // 2
        self.file_list = [self.file_list[idx:idx + self.sequence_size] for idx in
                          range(len(self.file_list) - self.sequence_size + 1)]
        self.video_names = [file_path_list[self.interval].split('/')[-2] for file_path_list in self.file_list]
        self.image_names = [file_path_list[self.interval].split('/')[-1].split('.')[0] for file_path_list in self.file_list]
        self.gt_name = None

    def load_image(self, data_list):
        central_stack = Image.open(data_list[self.interval]).convert('RGB')
        gt = None
        side_stack = [Image.open(data_list[idx]).convert('RGB') for idx in
                      range(len(data_list)) if idx != self.interval]

        return central_stack, gt, side_stack
