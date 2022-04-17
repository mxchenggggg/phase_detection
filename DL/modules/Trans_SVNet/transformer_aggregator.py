from pytorch_lightning import LightningModule
from torch import optim
import configargparse
import torchmetrics
import torch
import pickle
import os
import numpy as np


class TransSVNetTransformerAggregator(LightningModule):
    def __init__(self, hparams, model) -> None:
        super().__init__()

        self.hprms = hparams
        self.model = model

        self.ce_loss = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(hparams.class_weights).float())

        self.init_metrics()

    def init_metrics(self):
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hprms.learning_rate)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer]  # , [scheduler]

    def forward(self, spatial_features, temporal_features):
        output = self.model.forward(spatial_features, temporal_features)
        return output

    def loss(self, p_phase, labels_phase):
        loss = self.ce_loss(p_phase, labels_phase)
        return loss

    def training_step(self, batch, batch_idx):
        spatial_features, temporal_features, label = batch
        output = self.forward(spatial_features, temporal_features)
        output = output.squeeze()
        label = label.squeeze()

        self.train_acc(output, label)

        loss = self.loss(output, label)
        self.log("train_acc", self.train_acc,
                 on_epoch=True, on_step=True)
        self.log("loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=True)
        return {"loss": loss, "train_acc": self.train_acc}

    def validation_step(self, batch, batch_idx):
        spatial_features, temporal_features, label = batch
        output = self.forward(spatial_features, temporal_features)
        output = output.squeeze()
        label = label.squeeze()

        self.val_acc(output, label)

        loss = self.loss(output, label)
        self.log("val_acc", self.val_acc,
                 on_epoch=True, on_step=False)
        self.log("val_loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)
        return {"val_loss": loss, "val_acc": self.val_acc}

    def test_step(self, batch, batch_idx):
        spatial_features, temporal_features, label = batch
        output = self.forward(spatial_features, temporal_features)
        output = output.squeeze()
        label = label.squeeze()

        self.test_acc(output, label)

        loss = self.loss(output, label)
        self.log("test_acc", self.test_acc,
                 on_epoch=True, on_step=False)
        self.log("test_loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)
        return {"test_loss": loss, "outputs": (np.argmax(output.cpu().numpy(), axis=1), label.cpu().numpy())}

    def test_epoch_end(self, outputs) -> None:
        output_path = os.path.abspath(
            self.hprms.test_output_dir)
        output_path = os.path.join(output_path, self.hprms.name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_list = []
        for idx in range(len(outputs)):
            video_idx = self.hprms.test_video_indexes[idx]
            output_list.append((video_idx, outputs[idx]["outputs"][0], outputs[idx]["outputs"][1]))
        file_path = os.path.join(
            output_path, f"Trans_SVNet_outputs.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(output_list, f)

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        trans_svnet_transfomer_module_args = parser.add_argument_group(
            title='trans_svnet_sptial_module specific args options')
        trans_svnet_transfomer_module_args.add_argument(
            "--class_weights", type=float, nargs='+', required=True)
        trans_svnet_transfomer_module_args.add_argument(
            "--test_output_dir", type=str, required=True)
        return parser