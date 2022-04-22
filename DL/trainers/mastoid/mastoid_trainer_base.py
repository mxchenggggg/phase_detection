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


class MastoidTrainerBase:
    """ Trainer base class for Mastoidectomy 
        Surgical Phase Segmentation 

        This base class implemented following functionality for training/prediction:
            1. Load model, module, datasetmodule, transform from config file
            2. Parse config file arguments
            3. Interface for training and prediction

        Usage:
            Derive a trainer class from this base class (e.g. MyTrainer.py)
            or use this base class if no additional functionality needed:

                trainer = MyTrainer(default_config_file="DEFAULT_CONFIG_FILE_PATH")
                trainer() # perform training or predicition

            In CML, run:
                MyTrainer.py [-c|--config CONFIG_FILE_PATH] [-p|--predict] 
            Use flag -p or --predict for prediction mode;

            Prediction logics is handle by callbacks; refer to modules for more details
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
        self._get_classes_and_parse_args()

        # 4. setup output path
        self.hprms.log_output_path, self.hprms.pred_output_path, self.hprms.name = self._get_output_path_and_exp_name()

        # 5. loggers
        tb_logger = TensorBoardLogger(self.hprms.log_output_path, name='tb')
        wandb_logger = WandbLogger(
            name=self.hprms.name, project=self.hprms.project, entity="cis2mastoid")

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
                       ModelSummary(max_depth=-1)
                       ],
            resume_from_checkpoint=self.hprms.resume_from_checkpoint,
            num_sanity_val_steps=self.hprms.num_sanity_val_steps,
            log_every_n_steps=self.hprms.log_every_n_steps)
        trainer.fit(self.module, datamodule=self.datamodule)
        print(
            f"Best: {self.checkpoint_callback.best_model_score} | monitor: {self.checkpoint_callback.monitor} | path: {self.checkpoint_callback.best_model_path}"
            f"\nTesting..."
        )
        trainer.test(ckpt_path=self.checkpoint_callback.best_model_path,
                     datamodule=self.datamodule)

    def _predict(self) -> None:
        """ Prediction
        """
        trainer = Trainer(
            gpus=self.hprms.gpus, logger=False)

        trainer.predict(
            ckpt_path=self.hprms.resume_from_checkpoint,
            datamodule=self.datamodule, model=self.module)

    def _get_classes_and_parse_args(self) -> Any:
        """Get hyperparameters, model, module, datamodule and transform from config file
        """
        # get Classes
        dirs = ["modules", "modules", "modules", "models",
                "datasets", "datasets", "datasets"]
        names = ["module", "metrics_callback", "predictions_callback", "model",
                 "dataset", "datamodule", "transform"]
        for i in range(len(names)):
            path = f"{dirs[i]}.{getattr(self.hprms, names[i])}"
            attr_name = f"{''.join(x.capitalize()  for x in names[i].split('_'))}Class"
            setattr(self, attr_name, get_class_by_path(path))
            # add specific arguments to parser
            self.parser = getattr(
                self, attr_name).add_specific_args(
                self.parser)

        # parse arguments and print summary
        self.hprms = self.parser.parse_args()
        argparse_summary(self.hprms, self.parser)

        # get objects
        self.transform = self.TransformClass(self.hprms)
        self.datamodule = self.DatamoduleClass(
            self.hprms, self.DatasetClass, self.transform)
        self.model = self.ModelClass(self.hprms)
        self.module = self.ModuleClass(
            self.hprms, self.model, self.datamodule, self.MetricsCallbackClass,
            self.PredictionsCallbackClass)

    def _get_output_path_and_exp_name(self) -> Tuple[str, str]:
        """ Get output path and experiment name

        Returns:
            Tuple[str, str]: [output_path, exp_name]
                             output path for logs and checkpoints for training mode, 
                             and predicted features for prediciotn mode; experiment name
                             contians timestamp, module name, dataset name, and model name
        """
        # 1. experiment name
        if self.hprms.experiment_name:
            exp_name = self.hprms.experiment_name
        else:
            exp_name = (
                self.hprms.module.split(".")[-1] +
                "_" + self.hprms.dataset.split(".")
                [-1] + "_" + self.hprms.model.replace(".", "_"))
        date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
        exp_name = date_str + exp_name

        # 2. output path
        pred_output_path = os.path.abspath(
            self.hprms.prediction_output_path)
        if not os.path.exists(pred_output_path):
            os.makedirs(pred_output_path)

        log_output_path = os.path.join(
            os.path.abspath(self.hprms.logs_checkpoints_output_path),
            exp_name)

        return log_output_path, pred_output_path, exp_name

    def _get_checkpoint_callback(self) -> ModelCheckpoint:
        """ get checkpoint callback

        Returns:
            ModelCheckpoint: checkpoint callback
        """
        return ModelCheckpoint(
            dirpath=f"{self.hprms.log_output_path}/checkpoints/",
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
