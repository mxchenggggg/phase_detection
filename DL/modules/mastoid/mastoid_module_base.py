from pytorch_lightning import LightningModule
import torch
import configargparse
from typing import Tuple, Dict, List
from datasets.mastoid.mastoid_datamodule import MastoidDataModule
from torchmetrics import MetricCollection
from modules.mastoid.mastoid_metrics_callback_base import MastoidMetricsCallbackBase
from torchmetrics import ConfusionMatrix


class MastoidModuleBase(LightningModule):
    """ Mastoid module base class

    Args:
        LightningModule (_type_): _description_
    """

    def __init__(self, hparams, model: torch.nn.modules,
                 datamodule: MastoidDataModule, metrics_callback_class,
                 predictions_callback_class) -> None:
        super().__init__()
        # hyperparameters
        self.hprms = hparams

        # pointer to model
        self.model = model

        # pointer to datamodule
        self.datamodule = datamodule

        # cross entropy loss
        self.ce_loss = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(hparams.class_weights).float())

        # predictions outputs
        self.predictions_outputs = []

        # metric callback class
        self.metrics_callback_class = metrics_callback_class

        # prediction callback class
        self.predictions_callback_class = predictions_callback_class

        # initial metrics
        self.init_training_metrics()
        if hparams.predict:
            self.init_prediction_per_class_metrics()

    def init_training_metrics(self):
        # calling static method of metric callback class
        metric_list = []
        for _, metric_class in self.metrics_callback_class.get_metric_classes().items():
            # average of per-class metric
            metric_list.append(
                metric_class(
                    num_classes=self.hprms.num_classes,
                    average='macro'))

        # initialize metrics
        metrics = MetricCollection(metric_list)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def init_prediction_per_class_metrics(self):
        # calling static method of metric callback base class
        metric_list = []
        self.pred_metric_names = []
        # for metric_name, metric_class in MastoidMetricsCallbackBase.get_metric_classes().items():
        for metric_name, metric_class in self.metrics_callback_class.get_metric_classes().items():
            self.pred_metric_names.append(metric_name)
            metric_list.append(
                metric_class(
                    num_classes=self.hprms.num_classes,
                    average='none'))

        # initialize metrics
        metrics = MetricCollection(metric_list)
        self.pred_metrics_by_class = metrics.clone(
            prefix="pred_", postfix="_by_class")
        self.pred_cm = ConfusionMatrix(
            num_classes=self.hprms.num_classes)

    def configure_callbacks(self):
        # metric callback
        self.metrics_callback = self.metrics_callback_class(self.hprms)
        callback_list = [self.metrics_callback]

        # prediction callback
        if self.hprms.predict:
            self.prediction_callback = self.predictions_callback_class(
                self.hprms)
            callback_list.append(self.prediction_callback)

        return callback_list

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hprms.learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer]  # , [scheduler]

    def forward(self, batch: torch.Tensor) -> Dict:
        """ forward

        Args:
            batch (torch.Tensor): batch of data

        Returns:
            {"preds": torch.Tensor, "targets": torch.Tensor ...}
            model.forward should return a dictionary containing at least preds and targets
        """
        outputs = self.model.forward(batch)
        outputs["preds"] = outputs["preds"].squeeze()
        outputs["targets"] = outputs["targets"].squeeze()
        return outputs

    def loss(self, fwd_outputs):
        loss = self.ce_loss(fwd_outputs["preds"], fwd_outputs["targets"])
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self._forward_and_loss(batch, batch_idx)
        for key, val in outputs.items():
            if key != "loss":
                outputs[key] = val.detach()
        return outputs

    def validation_step(self, batch, batch_idx):
        return self._forward_and_loss(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._forward_and_loss(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self.predictions_outputs.append(outputs)
        return outputs

    def _forward_and_loss(
            self, batch: torch.Tensor, batch_idx: int) -> Dict:
        """ Forward and calculate loss

        Args:
            batch (torch.Tensor): data batch
            batch_idx (int): batch index

        Returns:
            {"preds": torch.Tensor, "targets": torch.Tensor, "loss" : torch.Tensor ...}
        """
        outputs = self.forward(batch)

        # calculate loss
        outputs["loss"] = self.loss(outputs)

        return outputs

    @ staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_module_args = parser.add_argument_group(
            title='trans_svnet_sptial_module specific args options')
        mastoid_module_args.add_argument(
            "--class_weights", type=float, nargs='+', required=True)
        mastoid_module_args.add_argument(
            "--class_names", type=str, nargs='+', required=True)
        return parser
