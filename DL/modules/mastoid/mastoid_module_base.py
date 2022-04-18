from pytorch_lightning import LightningModule
import torch
import configargparse
from typing import Tuple, Dict, List
from datasets.mastoid.mastoid_datamodule import MastoidDataModule
from modules.mastoid.mastoid_metrics_callback_base import MastoidMetricsCallbackBase


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
        metric_classes = self.metrics_callback_class.get_metric_classes()

        for stage in ["train", "val", "test"]:
            setattr(self, f'{stage}_metric_names', [])
            for name, metric_class in metric_classes.items():
                attr_name = f'{stage}_{name}'
                getattr(self, f'{stage}_metric_names').append(attr_name)
                setattr(self, f'{stage}_{name}', metric_class())

    def init_prediction_per_class_metrics(self):
        metric_classes = MastoidMetricsCallbackBase.get_metric_classes()
        setattr(self, 'pred_metric_names', [])
        for name, metric_class in metric_classes.items():
            getattr(self, 'pred_metric_names').append(name)
            setattr(self, f'pred_{name}_by_class', metric_class(
                num_classes=self.hprms.out_features, average='none'))
            setattr(self, f'pred_{name}_all', metric_class())

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
        loss = self.loss(outputs)
        outputs["loss"] = loss
        
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
