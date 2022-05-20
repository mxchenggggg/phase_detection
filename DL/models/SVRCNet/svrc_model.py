import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class ResLSTM(nn.Module):
    def __init__(self, hparams):
        self.num_classes = hparams.out_features  # 7 in cholec, 3 in Masto
        self.sequence_length = hparams.sequence_length
        self.batch_size = hparams.batch_size
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width
        super(ResLSTM, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Create the shared model (ResNet50)
        self.share = nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(
            input_size=2048, hidden_size=512, num_layers=1, batch_first=True
        )
        # add a FC layer to reduce from 512 to 128 first
        self.fc = nn.Linear(512, self.num_classes)  # fully connected 1

        # self.fc = nn.Linear(128, self.num_classes)
        # self.relu = nn.ReLU()

        nn.init.xavier_normal(self.lstm.all_weights[0][0])
        nn.init.xavier_normal(self.lstm.all_weights[0][1])
        nn.init.xavier_uniform(self.fc.weight)

    def forward(self, batch):
        x, targets = batch

        # Input size [Batch, sequence_len, 3, H, W] i.e [25,4,3,225,400]
        # First let CNN precess it one by one [25*4,3,255,400]
        curr_batch_size = x.shape[0]

        x = x.flatten(0, 1)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(curr_batch_size, self.sequence_length, 2048)

        self.lstm.flatten_parameters()
        preds, _ = self.lstm(x)
        preds = preds.contiguous().view(-1, 512)
        preds = self.fc(preds)

        return {"preds": preds, "targets": targets.flatten(0, 1)}

    @staticmethod
    def add_specific_args(parser):  # pragma: no cover
        reslstm_specific_args = parser.add_argument_group(
            title='reslstm specific args options')
        reslstm_specific_args.add_argument("--pretrained",
                                           action="store_true",
                                           help="pretrained on imagenet")
        reslstm_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=128)
        return parser
