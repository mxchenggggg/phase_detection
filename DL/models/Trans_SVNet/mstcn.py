# implementation adapted from:
# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MultiStageModel(nn.Module):
    def __init__(self, hparams):
        self.num_stages = hparams.mstcn_stages  # 4 #2
        self.num_layers = hparams.mstcn_layers  # 10  #5
        self.num_f_maps = hparams.mstcn_f_maps  # 64 #64
        self.dim = hparams.mstcn_f_dim  # 2048 # 2048
        self.num_classes = hparams.out_features  # 7
        self.causal_conv = hparams.mstcn_causal_conv
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False

    def forward(self, batch):
        spatial_features, targets = batch
        # transpose for 1D convolution
        # (batch_size, spatial_feature_dim, vid_len)
        spatial_features = spatial_features.transpose(2, 1)

        # (batch_size, num_classes, vid_len)
        prev_stage_preds = self.stage1(spatial_features)
        stage_preds = prev_stage_preds.unsqueeze(0)
        for s in self.stages:
            preds = s(F.softmax(prev_stage_preds, dim=1))
            stage_preds = torch.cat([stage_preds, preds.unsqueeze(0)], dim=0)

        # stage_preds: (num_stages, batch_size, num_classes, vid_len)
        stage_preds = F.softmax(stage_preds, dim = 2)
        # stage_preds: (num_stages, batch_size, vid_len, num_classes)
        stage_preds = stage_preds.transpose(2, 3)
        
        # use last stage outputs as preds
        outputs = {
            "preds": stage_preds[-1],
            "targets": targets, "spatial_features": spatial_features.transpose(1, 2).squeeze(),
            "stage_preds": stage_preds}
        return outputs


    @staticmethod
    def add_specific_args(parser):  # pragma: no cover
        mstcn_reg_model_specific_args = parser.add_argument_group(
            title='mstcn reg specific args options')
        mstcn_reg_model_specific_args.add_argument("--mstcn_stages",
                                                   default=4,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_layers",
                                                   default=10,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_maps",
                                                   default=64,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_dim",
                                                   default=2048,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_causal_conv",
                                                   action='store_true')
        return parser


class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes


class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class DilatedSmoothLayer(nn.Module):
    def __init__(self, causal_conv=True):
        super(DilatedSmoothLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation1 = 1
        self.dilation2 = 5
        self.kernel_size = 5
        if self.causal_conv:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2 * 2,
                                           dilation=self.dilation2)

        else:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2,
                                           dilation=self.dilation2)
        self.conv_1x1 = nn.Conv1d(7, 7, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x1 = self.conv_dilated1(x)
        x1 = self.conv_dilated2(x1[:, :, :-4])
        out = F.relu(x1)
        if self.causal_conv:
            out = out[:, :, :-((self.dilation2 * 2) * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)
