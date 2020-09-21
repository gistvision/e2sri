import torch
import torch.nn
from .base import ConvBlock, DeconvBlock, ResnetBlock, UpBlock, DownBlock

BLOCK_PARAMS = {
    2: {
        'kernel_size': 6,
        'stride': 2,
        'padding': 2,
    },
    4: {
        'kernel_size': 8,
        'stride': 4,
        'padding': 2,
    },
    8: {
        'kernel_size': 12,
        'stride': 8,
        'padding': 2,
    }
}


class SRNet(torch.nn.Module):
    def __init__(self, cfg):
        super(SRNet, self).__init__()
        center_in_channels = cfg.MODEL.IN_CHANNELS
        side_in_channels = cfg.MODEL.IN_CHANNELS * 2 + cfg.MODEL.FLOW_CHANNELS
        out_channels = cfg.MODEL.OUT_CHANNELS
        base_1_channels = cfg.MODEL.BASE_1_CHANNELS
        base_2_channels = cfg.MODEL.BASE_2_CHANNELS
        sequence_size = cfg.SEQUENCE_SIZE

        # EFR
        # central stack
        self.center_event_rectifier = ConvBlock(in_channels=center_in_channels,
                                                out_channels=base_1_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                activation='prelu')
        # other stacks
        self.side_event_rectifier = ConvBlock(in_channels=side_in_channels,
                                              out_channels=base_1_channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              activation='prelu')

        # SRNet
        self.rnet_a = RNet_A(cfg)
        self.rnet_b = RNet_B(cfg)
        self.rnet_c = RNet_C(cfg)
        self.rnet_d = RNet_D(cfg)

        # Mixer (mixes intermediate intensity outputs for final reconstruvtion)
        self.mixer = ConvBlock(in_channels=(sequence_size - 1) * base_2_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, central_stack, side_stack_list, flow_list):
        # EFR
        # central stack feature rectifier without optical flow
        state = self.center_event_rectifier(central_stack)
        # feature rectifier with optical flow for the sequence of stacks except central stack
        # eaxh stack is compared to central stack for calculating optical flow
        rectified_event_list = []
        for side_stack, flow in zip(side_stack_list, flow_list):
            rectified_event_list.append(self.side_event_rectifier(torch.cat((central_stack, side_stack, flow), 1)))

        # SRNet
        intermediate_images = []
        for rectified_event in rectified_event_list:
            rnet_a_out = self.rnet_a(state)
            rnet_c_out = self.rnet_c(rectified_event)

            e = rnet_a_out - rnet_c_out
            rnet_b_out = self.rnet_b(e)

            hidden_state = rnet_a_out + rnet_b_out
            intermediate_images.append(hidden_state)
            state = self.rnet_d(hidden_state)

        # Mix
        # Final output intensity image reconstruction by mixing all intermediate outputs
        mix = torch.cat(intermediate_images, 1)
        output = self.mixer(mix)

        return output


class RNet_A(torch.nn.Module):
    def __init__(self, cfg):
        super(RNet_A, self).__init__()
        stage = cfg.MODEL.SRNET_A.STAGE
        scale_factor = cfg.SCALE_FACTOR
        base_1_channels = cfg.MODEL.BASE_1_CHANNELS
        base_2_channels = cfg.MODEL.BASE_2_CHANNELS

        self.rectified_event_feature = ConvBlock(in_channels=base_1_channels,
                                                 out_channels=base_2_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 activation='prelu')
        # Hour-Glass (increase-decrease scale)
        self.HG_block = self.make_HG_block(stage=stage,
                                           num_channels=base_2_channels,
                                           scale_factor=scale_factor)
        # Initial HR recontstruction from stack
        self.union = ConvBlock(in_channels=base_2_channels * stage,
                               out_channels=base_2_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

    def forward(self, x):
        # input to RNet-A is rectified events
        # we make features from the rectified events (ref)
        ref = self.rectified_event_feature(x)
        # hour_glass stages increaseing/decreasing the resolution
        hg_out_list = []
        for block in self.HG_block:
            ref = block(ref)
            hg_out_list.append(ref)
        # Rnet-A output
        rnet_a_out = self.union(torch.cat(hg_out_list, 1))
        return rnet_a_out

    @staticmethod
    def make_HG_block(stage, num_channels, scale_factor):
        HG_block = []
        if stage == 0:
            return HG_block

        kernel_size = BLOCK_PARAMS[scale_factor]['kernel_size']
        stride = BLOCK_PARAMS[scale_factor]['stride']
        padding = BLOCK_PARAMS[scale_factor]['padding']
        # First HG_block only increase
        HG_block.append(UpBlock(num_channels=num_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                activation='prelu'))

        # Other HG_block decrease-increase
        for _ in range(stage - 1):
            HG_block.append(torch.nn.Sequential(
                DownBlock(num_channels=num_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          activation='prelu'),
                UpBlock(num_channels=num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        activation='prelu')
            ))

        HG_block = torch.nn.ModuleList(HG_block)
        return HG_block


class RNet_B(torch.nn.Module):
    def __init__(self, cfg):
        super(RNet_B, self).__init__()
        stage = cfg.MODEL.SRNET_B.STAGE
        base_2_channels = cfg.MODEL.BASE_2_CHANNELS

        modules = [ResnetBlock(num_channels=base_2_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               activation='prelu') for _ in range(stage)]
        modules.append(ConvBlock(in_channels=base_2_channels,
                                 out_channels=base_2_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 activation='prelu'))
        self.rnet_b = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.rnet_b(x)


class RNet_C(torch.nn.Module):
    def __init__(self, cfg):
        super(RNet_C, self).__init__()
        stage = cfg.MODEL.SRNET_C.STAGE
        base_1_channels = cfg.MODEL.BASE_1_CHANNELS
        base_2_channels = cfg.MODEL.BASE_2_CHANNELS
        scale_factor = cfg.SCALE_FACTOR
        kernel_size = BLOCK_PARAMS[scale_factor]['kernel_size']
        stride = BLOCK_PARAMS[scale_factor]['stride']
        padding = BLOCK_PARAMS[scale_factor]['padding']

        modules = [ResnetBlock(num_channels=base_1_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               activation='prelu') for _ in range(stage)]
        modules.append(DeconvBlock(in_channels=base_1_channels,
                                   out_channels=base_2_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   activation='prelu'))
        self.rnet_c = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.rnet_c(x)


class RNet_D(torch.nn.Module):
    def __init__(self, cfg):
        super(RNet_D, self).__init__()
        stage = cfg.MODEL.SRNET_D.STAGE
        base_1_channels = cfg.MODEL.BASE_1_CHANNELS
        base_2_channels = cfg.MODEL.BASE_2_CHANNELS
        scale_factor = cfg.SCALE_FACTOR
        kernel_size = BLOCK_PARAMS[scale_factor]['kernel_size']
        stride = BLOCK_PARAMS[scale_factor]['stride']
        padding = BLOCK_PARAMS[scale_factor]['padding']

        modules = [ResnetBlock(num_channels=base_2_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               activation='prelu') for _ in range(stage)]
        modules.append(ConvBlock(in_channels=base_2_channels,
                                 out_channels=base_1_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 activation='prelu'))
        self.rnet_d = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.rnet_d(x)
