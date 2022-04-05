import configargparse
import os
from pathlib import Path
from typing import Type
from datetime import datetime

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary

from utils.configargparse_arguments import build_configargparser
from utils.utils import (
    argparse_summary,
    get_class_by_path,
)

class MastoidTrainer:
    def __init__(self) -> None:
        # 1. initialize argument parser
        self.parser = configargparse.ArgParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser)
        self.parser.add_argument(
            '-c', is_config_file_arg=True, help='config file path')
        self.parser.add_argument(
            '-p', action='store_true', help="prediction mode flag")

        # 2. build argument parser
        self.parser, self.hprms = build_configargparser(self.parser)

        # 3. load classes from config file and add specific args
        self._get_classes_and_add_specific_args()
        argparse_summary(self.hprms, self.parser)

        # 4. setup output path
        self._set_output_path()

        # 5. loggers
        tb_logger = TensorBoardLogger(self.hprms.output_path, name='tb')
        wandb_logger = WandbLogger(name=self.hprms.name, project="transsvnet")
        self.loggers = [tb_logger, wandb_logger]

    def __call__(self) -> None:
        if self.hprms.p:
            # prediction mode
            print("prediction")
            trainer = Trainer(
                gpus=self.hprms.gpus, logger=self.loggers,
                resume_from_checkpoint=self.hprms.resume_from_checkpoint)
            predictions = trainer.predict(
                self.module, datamodule=self.datamodule)
            self.datamodule.save_predictions(predictions)
        else:
            self._set_checkpoint_callback()
            self._set_early_stop_callback()
            trainer = Trainer(
                gpus=self.hprms.gpus, logger=self.loggers,
                min_epochs=self.hprms.min_epochs,
                max_epochs=self.hprms.max_epochs,
                callbacks=[self.checkpoint_callback,
                           self.early_stop_callback,
                           ModelSummary(max_depth=-1)],
                resume_from_checkpoint=self.hprms.resume_from_checkpoint)
            trainer.fit(self.module, datamodule=self.datamodule)
            print(
                f"Best: {self.checkpoint_callback.best_model_score} | monitor: {self.checkpoint_callback.monitor} | path: {self.checkpoint_callback.best_model_path}"
                f"\nTesting..."
            )
            trainer.test(ckpt_path=self.checkpoint_callback.best_model_path,
                         datamodule=self.datamodule)

    def _get_classes_and_add_specific_args(self) -> None:
        """ Get ModuleClass, ModelClass, DatasetClass, DatamoduleClass,
            and TransformClas from config file, and add class-specific args
        """
        dirs = ["modules", "models", "datasets", "datasets", "datasets"]
        names = ["module", "model", "dataset", "datamodule", "transform"]
        for i in range(len(names)):
            path = f"{dirs[i]}.{getattr(self.hprms, names[i])}"
            attr_name = f"{names[i].capitalize()}Class"
            setattr(self, attr_name, get_class_by_path(path))
            getattr(self, attr_name).add_specific_args(self.parser)

        # parse specifc arguments
        self.hprms = self.parser.parse_args()

        # initialize class
        self.transform = self.TransformClass(self.hprms)
        self.datamodule = self.DatamoduleClass(
            self.hprms, self.DatasetClass, self.transform)
        self.model = self.ModelClass(self.hprms)
        self.module = self.ModuleClass(self.hprms, self.model)

    def _set_output_path(self) -> None:
        """ Setup output path for logs and checkpoints
        """
        exp_name = (
            self.hprms.module.split(".")[-1] + "_" + self.hprms.dataset.split(".")
            [-1] + "_" + self.hprms.model.replace(".", "_"))
        date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
        self.hprms.name = date_str + exp_name
        self.hprms.output_path = os.path.join(
            os.path.abspath(self.hprms.output_path), self.hprms.name)

    def _set_checkpoint_callback(self) -> None:
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=f"{self.hprms.output_path}/checkpoints/",
            save_top_k=self.hprms.save_top_k,
            verbose=True,
            monitor=self.hprms.early_stopping_metric,
            mode=self.hprms.early_stopping_metric_mode,
            filename=f'{{epoch}}-{{{self.hprms.early_stopping_metric}:.2f}}'
        )

    def _set_early_stop_callback(self) -> None:
        self.early_stop_callback = EarlyStopping(
            monitor=self.hprms.early_stopping_metric,
            mode=self.hprms.early_stopping_metric_mode,
            min_delta=0.00,
            patience=3,
        )


if __name__ == "__main__":
    trainer = MastoidTrainer()
    trainer()
