import torch.nn as nn
from torchvision import models

from models.WILDCAT.pooling import WildcatPool2d, ClassWisePool


class ResNetWSL50(nn.Module):

    def __init__(self, hparams):
        super(ResNetWSL50, self).__init__()
        self.dense = hparams.dense

        model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(
                num_features, hparams.num_classes * hparams.num_maps,
                kernel_size=1, stride=1, padding=0, bias=True))

        self.class_wise_pooling = ClassWisePool(hparams.num_maps)
        self.spatial_pooling = WildcatPool2d(
            hparams.kmax, hparams.kmin, hparams.alpha)

    def forward(self, batch):
        x, targets = batch
        features = self.features(x)
        features = self.classifier(features)
        maps = self.class_wise_pooling(features)
        preds = self.spatial_pooling(maps)
        return {"preds": preds, "targets": targets, "maps": maps}

    @staticmethod
    def add_specific_args(parser):  # pragma: no cover
        wildcat50_args = parser.add_argument_group(
            title='WILDCAT50 specific args options')
        wildcat50_args.add_argument("--kmax", default=1, type=int)
        wildcat50_args.add_argument("--kmin", default=1, type=int)
        wildcat50_args.add_argument("--alpha", default=0.6, type=float)
        wildcat50_args.add_argument("--num_maps", default=4, type=int)
        wildcat50_args.add_argument("--dense", default=False, type=bool)
        return parser
