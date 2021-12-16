import os
import argparse

import utils
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
parser.add_argument('--config_path', type=str, help='Select the training config file')
parser.add_argument('--data_dir', type=str, help='Training data directory')
parser.add_argument('--save_dir', type=str, help='Location to save checkpoint models')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for parallel threads')

args = parser.parse_args()

# Load Config File
cfg = utils.cfg
cfg.merge_from_file(args.config_path)
cfg.freeze()

# Load Dataset
train_set = datasets.SRDataset(root=args.data_dir, cfg=cfg, is_train=True)
train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers,
                          batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
test_set = datasets.SRDataset(root=args.data_dir, cfg=cfg, is_train=False)
test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers,
                         batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)

# Build Model
model = models.SRNet(cfg)
model = torch.nn.DataParallel(model).cuda()

l1_loss_function = torch.nn.L1Loss()
LPIPS_loss_function = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [cfg.TRAIN.EPOCH // 2, cfg.TRAIN.EPOCH * 3 // 4], gamma=0.1)

# Train!
for epoch in range(cfg.TRAIN.EPOCH):
    epoch_loss = 0.0
    model.train()
    for batch in tqdm(train_loader):
        central_stack = batch['central_stack'].cuda()
        gt = batch['gt'].cuda()
        side_stack = [data.cuda() for data in batch['side_stack']]
        flow = [data.cuda() for data in batch['flow']]

        optimizer.zero_grad()
        prediction = model(central_stack, side_stack, flow)

        l1_loss = l1_loss_function(prediction, gt)
        LPIPS_loss = LPIPS_loss_function(prediction, gt, normalize=True).mean()
        loss = l1_loss + (LPIPS_loss / 3.0)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print('Epoch %d Complete: Avg. Loss: %.4lf | lr: %.4e' % (
    epoch + 1, epoch_loss, scheduler.optimizer.param_groups[0]['lr']))
    scheduler.step()

    # Save Model
    torch.save({'config': cfg, 'model': model.state_dict()}, os.path.join(args.save_dir, 'checkpoint.pth'))

# Test!
with torch.no_grad():
    model.eval()
    for batch in tqdm(test_loader):
        central_stack = batch['central_stack'].cuda()
        gt = batch['gt'].cuda()
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
