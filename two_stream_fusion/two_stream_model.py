
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchsummary import summary

import numpy as np

from copy import deepcopy
import os


def gaussian_kernel_3D():
    x, y, z = np.meshgrid(np.linspace(-1,1,3), np.linspace(-1,1,3), np.linspace(-1,1,3))
    sigma= 1.0
    g = np.exp(-( (x*x+y*y+z*z) / ( 2.0 * sigma**2 ) ) ) / (np.sqrt(2. * np.pi))**3
    return torch.from_numpy(g)

class TwoStreamFusion(nn.Module):
    def __init__(self, rgb_frames, opf_frames, h, w, num_classes, dropout = 0.1, freeze_base=False, backbone_name = 'vgg16'):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.rgb_frames = rgb_frames
        self.opf_channels = 2 * opf_frames
        self.h, self.w = h, w

        pretrained = eval(f'{backbone_name}(pretrained=True)')
        
        self.spat_features = deepcopy(pretrained.features)

        self.temp_features = deepcopy(pretrained.features)
        self.temp_features[0] = nn.Conv2d(
            self.opf_channels, 64, kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1))

        # Average across RGB channels and initialize first conv of temperal feature network.
        first_conv_weight_rgb_avg = \
            torch.mean(pretrained.features[0].weight.data, dim=1, keepdim=True)
        self.temp_features[0].weight.data = first_conv_weight_rgb_avg.repeat(
            1, 20, 1, 1)

        if freeze_base:
            for param in self.spat_features.parameters():
                param.requires_grad = False
            for param in self.temp_features.parameters():
                param.requires_grad = False

        self.fusion = nn.Sequential(
            nn.Conv3d(
                1024, 512, 3, stride=1, padding=1, dilation=1, bias=True),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2))

        # nn.init.zeros_(self.fusion[0].weight)
        g = gaussian_kernel_3D()
        for i in range(512):
            with torch.no_grad():
                self.fusion[0].weight[i][i * 2].copy_(g)
                self.fusion[0].weight[i][i * 2 + 1].copy_(g * 3.)

        self.fc = nn.Sequential(
            nn.Linear(18432, 2048),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # N, T, C, H, W
        rgb_frames, opf_frames = x

        # N * T, C, H, W
        rgb_frames = rgb_frames.view(-1, 3, self.h, self.w)
        # N * T, C, H', W'
        rgb_out = self.spat_features(rgb_frames)
        # N, T, C, H', W'
        rgb_out = rgb_out.view(-1, self.rgb_frames, *(rgb_out.size()[1:]))
        # N, C, T, H', W'
        rgb_out = torch.transpose(rgb_out, 2, 1)

        # N * T, C, H, W
        opf_frames = opf_frames.view(-1, self.opf_channels, self.h, self.w)
        # N * T, C, H', W'
        opf_out = self.temp_features(opf_frames)
        # N, T, C, H', W'
        opf_out = opf_out.view(-1, self.rgb_frames, *(opf_out.size()[1:]))
        # N, C, T, H', W'
        opf_out = torch.transpose(opf_out, 2, 1)

        # N, 2 * C, T, H', W'
        concat_out = torch.stack([rgb_out, opf_out], dim=2).flatten(1, 2)
        fusion_out = self.fusion(concat_out)
        fusion_out = fusion_out.view(fusion_out.size(0), -1)

        out = self.fc(fusion_out)
        return out


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    batch_size = 16
    rgb_frames, opf_frames = 5, 10
    H, W = 255, 400
    num_classes = 5

    model = TwoStreamFusion(rgb_frames, opf_frames, H, W, num_classes, backbone_name='vgg19')
    model.cuda()
    model = nn.DataParallel(model)

    rgb_input_shape = (batch_size, rgb_frames, 3, H, W)
    opf_input_shape = (batch_size, rgb_frames, 2 * opf_frames, H, W)

    rgb_frames = torch.from_numpy(np.zeros(rgb_input_shape, dtype=np.float32))
    opf_frames = torch.from_numpy(np.zeros(opf_input_shape, dtype=np.float32))

    rgb_frames = Variable(rgb_frames.cuda())
    opf_frames = Variable(opf_frames.cuda())
    intput = (rgb_frames, opf_frames)
    print("input shape", rgb_frames.shape, opf_frames.shape)

    out = model(intput)
    print("output shape", out.shape)
