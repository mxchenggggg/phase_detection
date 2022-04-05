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


def train(
        hparams, ModuleClass, ModelClass, DatasetClass,
        DataModuleClass: Type[LightningModule], TransformClass,
        logger):
    transform = TransformClass(hparams)

    # load datamodule
    datamodule = DataModuleClass(
        hparams, DatasetClass, transform)

    # loda model
    model = ModelClass()

    # load module
    module = ModuleClass(hparams, model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.early_stopping_metric,
        mode='max',
        filename=f'{hparams.name}-{{epoch}}-{{{hparams.early_stopping_metric}:.2f}}'
    )
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        patience=3,
        mode='max')

    trainer = Trainer(
        gpus=hparams.gpus, logger=logger, min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback,
                   ModelSummary(max_depth=-1)],
        resume_from_checkpoint=hparams.resume_from_checkpoint)

    trainer.fit(module, datamodule=datamodule)
    print(
        f"Best: {checkpoint_callback.best_model_score} | monitor: {checkpoint_callback.monitor} | path: {checkpoint_callback.best_model_path}"
        f"\nTesting..."
    )

    trainer.test(ckpt_path=checkpoint_callback.best_model_path,
                 datamodule=datamodule)


def get_class_and_add_args(parser, hparams, dir: str, name: str):
    path = f"{dir}.{getattr(hparams, name)}"
    Class = get_class_by_path(path)
    return Class.add_specific_args(parser)


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', is_config_file_arg=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    ModuleClass = get_class_and_add_args(parser, hparams, "modules", "module")
    ModelClass = get_class_and_add_args(parser, hparams, "models", "model")
    DatasetClass = get_class_and_add_args(
        parser, hparams, "datasets", "dataset")
    DataModuleClass = get_class_and_add_args(
        parser, hparams, "datasets", "datamodule")
    TransformClass = get_class_and_add_args(
        parser, hparams, "datasets", "transform")

    hparams = parser.parse_args()
    argparse_summary(hparams, parser)

    exp_name = (
        hparams.module.split(".")[-1] + "_" + hparams.dataset.split(".")[-1] +
        "_" + hparams.model.replace(".", "_"))
    date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
    hparams.name = date_str + exp_name
    hparams.output_path = os.path.join(
        os.path.abspath(hparams.output_path), hparams.name)

    tb_logger = TensorBoardLogger(hparams.output_path, name='tb')
    wandb_logger = WandbLogger(name=hparams.name, project="transsvnet")

    loggers = [tb_logger, wandb_logger]

    train(hparams, ModuleClass, ModelClass,
          DatasetClass, DataModuleClass, TransformClass, loggers)
