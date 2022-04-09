from pytorch_lightning import LightningModule
from torch import optim
import configargparse
import torchmetrics
import torch


class TransSVNetTransformerAggregator(LightningModule):
    def __init__(self, hparams, model) -> None:
        super().__init__()

        self.hprams = hparams
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
                               lr=self.hprams.learning_rate)
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
        return {"test_loss": loss}

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        trans_svnet_transfomer_module_args = parser.add_argument_group(
            title='trans_svnet_sptial_module specific args options')
        trans_svnet_transfomer_module_args.add_argument(
            "--class_weights", type=float, nargs='+', required=True)
        return parser