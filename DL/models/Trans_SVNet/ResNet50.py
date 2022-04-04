import torch
import torch.nn as nn
from torchvision import models

class ResNet50TransSV(torch.nn.Module):
    def __init__(self):
        super(ResNet50TransSV, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512, 3))

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        y = self.fc(x)
        return y

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50_trans_svnet_args = parser.add_argument_group(
            title='resnet50_trans_svnet specific args options')
        return parser