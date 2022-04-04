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
        hparams, DatasetClass, transform.get_transform("train"))

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
        # prefix=hparams.name,
        filename=f'{hparams.name}-{{epoch}}-{{{hparams.early_stopping_metric}:.2f}}'
    )
    early_stop_callback = EarlyStopping(
        monitor=hparams.early_stopping_metric,
        min_delta=0.00,
        patience=3,
        mode='max')

    trainer = Trainer(
        gpus=hparams.gpus,
        logger=logger,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=-1)]
    )

    trainer.fit(module, datamodule=datamodule)
    print(
        f"Best: {checkpoint_callback.best_model_score} | monitor: {checkpoint_callback.monitor} | path: {checkpoint_callback.best_model_path}"
        f"\nTesting..."
    )

    trainer.test(ckpt_path=checkpoint_callback.best_model_path,
                 datamodule=datamodule)


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # module class
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)

    # model class
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)

    # dataset class
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)

    # datamodule class
    datamodule_path = f"datasets.{hparams.datamodule}"
    DataModuleClass = get_class_by_path(datamodule_path)
    parser = DataModuleClass.add_datamodule_specific_args(parser)

    # transform class
    transform_path = f"datasets.{hparams.transform}"
    TransformClass = get_class_by_path(transform_path)
    parser = TransformClass.add_transform_specific_args(parser)

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
    # wandb_logger = WandbLogger(name=hparams.name, project="transsvnet")

    # loggers = [tb_logger, wandb_logger]
    loggers = [tb_logger]

    train(hparams, ModuleClass, ModelClass,
          DatasetClass, DataModuleClass, TransformClass, loggers)
