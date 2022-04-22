from modules.mastoid.mastoid_module_base import MastoidModuleBase
from torchmetrics import Accuracy, Precision, Recall
from typing import Dict, Any, Optional, List
import pytorch_lightning as pl
import configargparse
import pandas as pd
import os
import pickle
import torch.nn.functional as F
import torch
import numpy as np


class MastoidPredictionsCallbackBase(pl.Callback):
    """ Mastoid predictions callback base class

        The prediction callback will be setup in mastoid_module and called at the 
        end of prediction by pytorch lightnings automatically.

        Operations in _on_prediction_end_operations will be performed with the
        callback is triggered. By default, it calls _eval_and_save_results which
        calculate and save both by-video and overall train/test/val evaluation results, 
        including Accuracy, Precision, Recall for all and per-class. 
        Evaluation resutls will be save as pkl and txt in predictions_output directory 
        setup in the config file

        This is intended to be a abstract class, derived class should implement the 
        function _split_predictions_outputs_by_videos which split predicion results by video

        To add more operations such as save prediction features, override the method
        _on_prediction_end_operations, notice that in the override method, you need to:
            1. call super()._on_prediction_end_operations(trainer, module) to calculate and save 
               per-video evaluations.
            2. super()._on_prediction_end_operations(trainer, module) returns outputs_by_videos, 
               which can be used for other operations.

    Args:
        pl (_type_): _description_
    """

    def __init__(self, hparms) -> None:
        self.hprms = hparms

    def on_predict_end(
            self, trainer: pl.Trainer, module: MastoidModuleBase) -> None:
        self._on_prediction_end_operations(trainer, module)

    def _on_prediction_end_operations(
            self, trainer: pl.Trainer, module: MastoidModuleBase) -> None:
        """ Operations performed at the end of predictions (e.g. save predicted features)

            Derive class should override this method to add more operations
            This base class split predictions and evaluate results by video
            call super()._on_prediction_end_operations(trainer, module) to obtain 
            outputs by video and generate per-video evaluation results
        Args:
            trainer (pl.Trainer): Trainer
            module (MastoidModuleBase): module

        Returns:
            Dict: Split predictions outputs by videos
        """
        return self._eval_and_save_results(module, module.predictions_outputs)

    def _split_predictions_outputs_by_videos(
            self, module: MastoidModuleBase, pred_outputs: List) -> Dict:
        """ Split predictions outputs by videos

        Args:
            module (MastoidModule): mastoid pytorch lightning module
            pred_outputs (List): list of batch results for predictions

        Raises:
            NotImplementedError: Derived class should implement this

        Returns:
            Dict: {video_idx : {"preds": tensor.Tensor, "targets" : tensor.Tensor ...} }
            A dictionary with video index as keys 
            and dictionaries with "preds" and "targets" as value
            "preds" and "targets" are required
        """
        raise NotImplementedError

    def _eval_and_save_results(self, module: MastoidModuleBase,
                               epoch_outputs: List) -> Dict:
        """ calculate and save both by-video and overall train/test/val 
            evaluation results including Accuracy, Precision, Recall 
            for both all and per-class. 

        Args:
            module (MastoidModuleBase): module
            epoch_outputs (List): list of predicion resutls for each epoch

        Returns:
            Dict: outputs_by_video returned by _split_predictions_outputs_by_videos
        """
        # split predictions output by videos
        outputs_by_videos = self._split_predictions_outputs_by_videos(
            module, epoch_outputs)

        # evaluation results data frames
        prediction_results_all_videos = {}

        # calculate per-class/all metric results for each video
        for vid_idx, outputs in outputs_by_videos.items():
            preds = F.softmax(outputs["preds"], dim=1)
            targets = outputs["targets"]
            outputs["eval_df"] = self._get_perclass_and_all_metric_eval_df(
                module, preds, targets)
            prediction_results_all_videos[vid_idx] = outputs

        # calculate per-class/all metric result for train/val/test
        prediction_results_all_videos["train_vid_idxes"] = module.datamodule.vid_idxes["train"]
        prediction_results_all_videos["val_vid_idxes"] = module.datamodule.vid_idxes["val"]
        prediction_results_all_videos["test_vid_idxes"] = module.datamodule.vid_idxes["test"]
        prediction_results_all_videos["pred_vid_idxes"] = module.datamodule.vid_idxes["pred"]
        for split in ["train", "val", "test"]:
            all_preds = []
            all_targets = []

            for vid_idx in module.datamodule.vid_idxes[split]:
                all_preds.append(outputs_by_videos[vid_idx]["preds"])
                all_targets.append(outputs_by_videos[vid_idx]["targets"])
            targets = torch.cat(all_targets)
            preds = F.softmax(torch.cat(all_preds), dim=1)
            
            df = self._get_perclass_and_all_metric_eval_df(
                module, preds, targets)
            prediction_results_all_videos[split] = {"eval_df": df}

        # save evaluation results

        # pickle file
        pred_results_file_pkl = os.path.join(
            os.path.abspath(self.hprms.pred_output_path),
            "prediction_results.pkl")
        with open(pred_results_file_pkl, "wb") as f:
            pickle.dump(prediction_results_all_videos, f)

        # text file
        pred_eval_result_file_txt = os.path.join(
            os.path.abspath(self.hprms.pred_output_path),
            "prediction_evaluation_results.txt")
        with open(pred_eval_result_file_txt, "w") as f:
            for key, results in prediction_results_all_videos.items():
                # print evaluation result dataframe to txt file
                df = results["eval_df"].astype(float)
                # video index
                if type(key) == int:
                    title = f'Video {vid_idx}\n'
                # train/test/val
                else:
                    title = f'{key} videos\n'
                f.write(title)
                df.to_csv(f, sep='\t', decimal='.', float_format="%.3f")
                f.write('\n')
        
        # outputs by videos returned by _split_predictions_outputs_by_videos
        return outputs_by_videos

    def _get_perclass_and_all_metric_eval_df(
            self, module: MastoidModuleBase, preds: torch.Tensor,
            targets: torch.Tensor) -> pd.DataFrame:
        """ Helper function for obtaining evaluation result Dataframe

        Args:
            module (MastoidModuleBase): module
            preds (torch.Tensor): predictions
            targets (torch.Tensor): targets

        Returns:
            pd.DataFrame: evaluation result
        """
        # class names
        class_names = module.hprms.class_names

        # pandans Dataframe to be returned
        df = pd.DataFrame(
            columns=class_names + ["all"],
            index=module.pred_metric_names)

        # calculate result by class
        result_by_class = module.pred_metrics_by_class(preds, targets)
        module.pred_metrics_by_class.reset()

        # iterate all metrics
        for metric_name in module.pred_metric_names:
            # add result per class to fg
            values_by_class = result_by_class[f'pred_{metric_name}_by_class']
            for i, class_name in enumerate(class_names):
                # iterate all classes
                val = values_by_class[i].item()
                df.loc[metric_name, class_name] = val

            df.loc[metric_name, "all"] = values_by_class.mean().item()
        return df

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_prediction_callback_args = parser.add_argument_group(
            title='trans_svnet_sptial_module specific args options')
        return parser
