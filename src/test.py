import os
import argparse

import models
import datasets

import torch.nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

cudnn.benchmark = True

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, help='Location to save checkpoint models')
parser.add_argument('--data_dir', type=str, help='Testing data directory')
parser.add_argument('--save_dir', type=str, help='Location to save output images')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for pytorch DataLoader')

args = parser.parse_args()

# Load Checkpoint
checkpoint = torch.load(args.checkpoint_path)

# Load Config File
cfg = checkpoint['config']

# Load Dataset
test_set = datasets.SingleFolderDataset(root=args.data_dir, cfg=cfg, is_train=False)
test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers,
                         batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)

# Build Model
model = models.SRNet(cfg)
model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint['model'])
model = model.cuda()

# Test!
with torch.no_grad():
    model.eval()
    for batch in tqdm(test_loader):
        central_stack = batch['central_stack'].cuda()
        side_stack = [data.cuda() for data in batch['side_stack']]
        flow = [data.cuda() for data in batch['flow']]
        video_name = batch['video_name']
        image_name = batch['image_name']

        prediction = model(central_stack, side_stack, flow)

        for idx in range(prediction.size(0)):
            cur_pred = prediction[idx].clamp(0.0, 1.0)
            cur_video_name = video_name[idx]
            cur_image_name = image_name[idx]

            generated_image = transforms.ToPILImage()(cur_pred.cpu()).convert('L')
            os.makedirs(os.path.join(args.save_dir, cur_video_name), exist_ok=True)
            generated_image.save(os.path.join(args.save_dir, cur_video_name, '%s.png' % cur_image_name))
