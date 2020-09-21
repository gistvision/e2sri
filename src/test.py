import os
import argparse

import utils
import models
import datasets

import torch.nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datetime import datetime

cudnn.benchmark = True

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Testing data directory')
parser.add_argument('--checkpoint_dir', type=str, help='Location to save checkpoint models')
parser.add_argument('--save_dir', type=str, help='Location to save output images')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for pytorch DataLoader')
parser.add_argument('--cpu_only', default=False, action='store_true', help='Set True for cpu only')

args = parser.parse_args()

# Check GPU
cuda = not args.cpu_only
device = 'cuda' if cuda else 'cpu'
if cuda and not torch.cuda.is_available():
    raise Exception("Missing GPU, please set --cpu_only to True for cpu only")

# Load Config File
cfg = utils.cfg
cfg.merge_from_file(os.path.join(args.checkpoint_dir, 'config.yaml'))
cfg.freeze()

# Load Dataset
test_set = datasets.SingleFolderDataset(root=args.data_dir, cfg=cfg, is_train=False)
test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers,
                         batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)

# Build Model
model = models.SRNet(cfg)
if cuda:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'weight', 'Final.pth')))
model = model.to(device)

# Test!
with torch.no_grad():
    model.eval()
    for iteration, batch in enumerate(test_loader):
        central_stacks = batch[0].to(device)
        side_stacks = [data.to(device) for data in batch[1]]
        flows = [data.to(device) for data in batch[2]]
        image_idxes = batch[3]

        predictions = model(central_stacks, side_stacks, flows)

        for idx in range(predictions.size(0)):
            prediction = predictions[idx].clamp(0.0, 1.0)
            image_idx = image_idxes[idx]

            generated_image = transforms.ToPILImage()(prediction.cpu()).convert('L')
            generated_image.save(os.path.join(args.save_dir, image_idx))
