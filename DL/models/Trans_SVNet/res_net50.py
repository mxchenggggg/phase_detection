import torch.nn as nn
from torchvision import models


class ResNet50TransSV(nn.Module):
    def __init__(self, hparams):
        super(ResNet50TransSV, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # replace final layer with number of labels
        self.model.fc = Identity()
        self.fc_phase = nn.Linear(2048, hparams.out_features)

    def forward(self, x):
        out_stem = self.model(x)
        phase = self.fc_phase(out_stem)
        return phase

    def get_spatial_feature(self, x):
        x = self.model.forward(x)
        x = x.view(-1, 2048)
        return x

    @staticmethod
    def add_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        return parser


#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
