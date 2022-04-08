import configargparse
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary

from utils.configargparse_arguments import build_configargparser
from utils.utils import (
    argparse_summary,
    get_class_by_path,
)

from typing import Optional, Tuple, Any
import warnings


class MastoidTrainerBase:
    """ Trainer base class for Mastoidectomy 
        Surgical Phase Segmentation 

        This base class implemented following functionality for training/prediction:
            1. Load model, module, datasetmodule, transform from config file
            2. Parse config file arguments
            3. Interface for training and prediction

        Usage:
            Derive a trainer class from this base class (e.g. MyTrainer.py):
                class MyTrainer(MastoidTrainerBase):
                    def _predict(self):
                        # prediction logic
                        ...

                trainer = MyTrainer(default_config_file="DEFAULT_CONFIG_FILE_PATH")
                trainer() # perform training or predicition

            In CML, run:
                MyTrainer.py [-c|--config CONFIG_FILE_PATH] [-p|--predict] 
            Use flag -p or --predict for prediction mode;
            _predict must be implemented for predicitno mode.
    """

    def __init__(self, default_config_file:  Optional[str] = None) -> None:
        # 1. initialize argument parser
        if default_config_file is None:
            self.parser = configargparse.ArgParser(
                config_file_parser_class=configargparse.YAMLConfigFileParser)
        else:
            self.parser = configargparse.ArgParser(
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                default_config_files=[default_config_file])
        self.parser.add_argument(
            '-c', '--config', is_config_file_arg=True, help='config file path')
        self.parser.add_argument(
            '-p', '--predict', action='store_true', help="prediction mode flag")

        # 2. build argument parser
        self.parser, self.hprms = build_configargparser(self.parser)

        # 3. load hyperparameters, classes from config file
        self.hprms, self.model, self.module, self.datamodule, self.transform = self._get_classes_and_parse_args()

        # 4. setup output path
        self.hprms.output_path, self.hprms.name = self._get_output_path_and_exp_name()

        # 5. loggers
        tb_logger = TensorBoardLogger(self.hprms.output_path, name='tb')
        wandb_logger = WandbLogger(name=self.hprms.name, project="transsvnet")
        self.loggers = [tb_logger, wandb_logger]

    def __call__(self) -> None:
        """ Trainer is a callbale object, MyTrainer() will trigger training or prediciton.
        """
        if self.hprms.predict:
            # prediction mode
            self._predict()
        else:
            # training mode
            self._train()

    def _train(self) -> None:
        """ Training
        """
        self.checkpoint_callback = self._get_checkpoint_callback()
        self.early_stop_callback = self._get_early_stop_callback()
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

    def _predict(self) -> None:
        """ Prediction
            Dereived class should implement this 
        """
        warnings.warn("Warinig: No prediction logic implemented")
        pass

    def _get_classes_and_parse_args(self) -> Any:
        """Get hyperparameters, model, module, datamodule and transform from config file
        """
        # get Classes
        dirs = ["modules", "models", "datasets", "datasets", "datasets"]
        names = ["module", "model", "dataset", "datamodule", "transform"]
        for i in range(len(names)):
            path = f"{dirs[i]}.{getattr(self.hprms, names[i])}"
            attr_name = f"{names[i].capitalize()}Class"
            setattr(self, attr_name, get_class_by_path(path))
            # add specific arguments to parser
            self.parser = getattr(
                self, attr_name).add_specific_args(
                self.parser)

        # parse arguments and print summary
        hprms = self.parser.parse_args()
        argparse_summary(hprms, self.parser)

        # get objects
        transform = self.TransformClass(hprms)
        datamodule = self.DatamoduleClass(hprms, self.DatasetClass, transform)
        model = self.ModelClass(hprms)
        module = self.ModuleClass(hprms, model)

        return hprms, model, module, datamodule, transform

    def _get_output_path_and_exp_name(self) -> Tuple[str, str]:
        """ Get output path and experiment name

        Returns:
            Tuple[str, str]: [output_path, exp_name]
                             output path for logs and checkpoints for training mode, 
                             and predicted features for prediciotn mode; experiment name
                             contians timestamp, module name, dataset name, and model name
        """
        # 1. experiment name
        exp_name = (
            self.hprms.module.split(".")[-1] + "_" + self.hprms.dataset.split(".")
            [-1] + "_" + self.hprms.model.replace(".", "_"))
        date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
        exp_name = date_str + exp_name

        # 2. output path
        if self.hprms.predict:
            # prediction mode
            output_path = os.path.abspath(
                self.hprms.prediction_output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        else:
            # training mode
            output_path = os.path.join(
                os.path.abspath(self.hprms.logs_checkpoints_output_path),
                exp_name)

        return output_path, exp_name

    def _get_checkpoint_callback(self) -> ModelCheckpoint:
        """ get checkpoint callback

        Returns:
            ModelCheckpoint: checkpoint callback
        """
        return ModelCheckpoint(
            dirpath=f"{self.hprms.output_path}/checkpoints/",
            save_top_k=self.hprms.save_top_k,
            verbose=True,
            monitor=self.hprms.early_stopping_metric,
            mode=self.hprms.early_stopping_metric_mode,
            filename=f'{{epoch}}-{{{self.hprms.early_stopping_metric}:.2f}}'
        )

    def _get_early_stop_callback(self) -> EarlyStopping:
        """ get early stop callback

        Returns:
            EarlyStopping: early stop callback
        """
        return EarlyStopping(
            monitor=self.hprms.early_stopping_metric,
            mode=self.hprms.early_stopping_metric_mode,
            min_delta=0.00,
            patience=3,
        )
