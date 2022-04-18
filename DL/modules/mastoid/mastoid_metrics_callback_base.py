from torchmetrics import Accuracy, Precision, Recall
from typing import Dict, Any, Optional, List
import pytorch_lightning as pl
import configargparse


class MastoidMetricsCallbackBase(pl.Callback):
    """ Mastoid Metrics Callback Base Class

        Base class implements metrics calculation and logging during 
        training/validation/testing.

        Pytorch lightning modules will call the static metod get_metric_classes
        and initialize all metrics objects so that all tentors are on correct device.

        Derived class from this base class should override the static method
        get_metric_classes to add additional metrics. Additional metrics will be logged
        during training/validation/testing as default ones.
    """
    def __init__(self, hparams) -> None:
        self.hprms = hparams

    @staticmethod
    def get_metric_classes():
        # default metrics
        return {"acc": Accuracy, "prec": Precision, "recall": Recall}

    ### on train/val/test batch end ###

    def on_train_batch_end(
            self, trainer: pl.Trainer, module: pl.LightningModule,
            outputs: Dict, batch: Any, batch_idx: int,
            unused: Optional[int] = 0) -> None:
        self._batch_eval_and_log(module, outputs, "train")

    def on_validation_batch_end(
            self, trainer: pl.Trainer, module: pl.LightningModule,
            outputs: Dict,
            batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self._batch_eval_and_log(module, outputs, "val")

    def on_test_batch_end(
            self, trainer: pl.Trainer, module: pl.LightningModule,
            outputs: Dict,
            batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self._batch_eval_and_log(module, outputs, "test")

    ### helper functions ###

    def _batch_eval_and_log(
            self, module: pl.LightningModule, outputs: Dict, stage: str) -> Dict:
        """ Calculate and log batch evaluations

        Args:
            module (MastoidModule): mastoid pytorch lightning module
            outputs (Dict): batch output {"preds": preds, "targets": targets, "loss": loss}
            stage (str): train/val/test

        Returns:
            Dict: preformance dictionary
        """
        performace_dict = {}
        for name in getattr(module, f'{stage}_metric_names'):
            getattr(module, name)(outputs["preds"], outputs["targets"])
            performace_dict[name] = getattr(module, name)
        performace_dict[f"{stage}_loss"] = outputs["loss"]
        module.log_dict(performace_dict, on_epoch=True)

        return performace_dict

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_metrics_callback_args = parser.add_argument_group(
            title='trans_svnet_sptial_module specific args options')
        return parser
