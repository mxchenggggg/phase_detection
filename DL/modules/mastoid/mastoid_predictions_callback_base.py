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
import matplotlib.pyplot as plt
import seaborn as sn


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
            eval_dfs = self._get_perclass_and_all_metric_eval_and_cm_df(
                module, preds, targets)
            for name, df in eval_dfs.items():
                outputs[name] = df
            prediction_results_all_videos[vid_idx] = outputs

        # calculate per-class/all metric result for train/val/test
        prediction_results_all_videos["vid_idxes"] = module.datamodule.vid_idxes
        for split in ["train", "val", "test"]:
            all_preds = []
            all_targets = []

            for vid_idx in module.datamodule.vid_idxes[split]:
                all_preds.append(outputs_by_videos[vid_idx]["preds"])
                all_targets.append(outputs_by_videos[vid_idx]["targets"])
            targets = torch.cat(all_targets)
            preds = F.softmax(torch.cat(all_preds), dim=1)

            prediction_results_all_videos[split] = self._get_perclass_and_all_metric_eval_and_cm_df(
                module, preds, targets)

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
                if "eval_df" in results:
                    # print evaluation result dataframe to txt file
                    eval_df = results["eval_df"].astype(float)
                    cm_df = results["cm_df"].astype(int)
                    cm_normalized_df = results["cm_normalized_df"].astype(float)

                    # video index
                    if type(key) == int:
                        title = f'Video_{key}'
                    # train/test/val
                    else:
                        title = f'{key}_videos'

                    self._plot_cm(cm_df, title, 'd')
                    self._plot_cm(cm_normalized_df, title + ' Normalized', '.3f')

                    eval_df_str = eval_df.to_string(float_format='%.3f')
                    cm_df_str = cm_df.to_string(float_format='%.3f')
                    cm_normalized_df_str = cm_normalized_df.to_string(
                        float_format='%.3f')

                    output_str = f'{title}\n{eval_df_str}\n'
                    output_str += f'Confusion Matrix\n{cm_df_str}\n'
                    output_str += f'Normalized Confusion Matrix\n{cm_normalized_df_str}\n'
                    f.write(output_str)
                    print(output_str)

        # outputs by videos returned by _split_predictions_outputs_by_videos
        return outputs_by_videos

    def _plot_cm(self, cm_df: pd.DataFrame, title, format):
        fig = plt.figure(figsize=(10, 7))
        sn.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt=format)
        plt.xlabel("Predicted Label", fontdict={'fontsize': 22})
        plt.ylabel("True Label", fontdict={'fontsize': 22})
        plt.title(f"{title} Confusion Matrix",
                  fontdict={'fontsize': 20})
        path = os.path.join(
            self.hprms.pred_output_path, f"{title}_cm.png")
        fig.savefig(path)
        plt.close(fig)

    def _get_perclass_and_all_metric_eval_and_cm_df(
            self, module: MastoidModuleBase, preds: torch.Tensor,
            targets: torch.Tensor) -> Dict:
        """ Helper function for obtaining evaluation result and confusion matrix Dataframe

        Args:
            module (MastoidModuleBase): module
            preds (torch.Tensor): predictions
            targets (torch.Tensor): targets

        Returns:
            Dict: {"eval_df": eval_df, "cm_df": cm_df,
                "cm_normalized_df": cm_normalized_df}
        """
        # class names
        class_names = module.hprms.class_names

        # pandans Dataframe to be returned
        eval_df = pd.DataFrame(
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
                eval_df.loc[metric_name, class_name] = val

            eval_df.loc[metric_name, "all"] = values_by_class.mean().item()

        cm_result = module.pred_cm(preds, targets)
        module.pred_cm.reset()
        cm_result = cm_result.cpu().numpy()
        cm_result_normalized = cm_result / np.sum(cm_result, axis=1)[:,np.newaxis]   
        cm_df = pd.DataFrame(cm_result, index=class_names, columns=class_names)
        cm_normalized_df = pd.DataFrame(
            cm_result_normalized, index=class_names, columns=class_names)
        return {"eval_df": eval_df, "cm_df": cm_df,
                "cm_normalized_df": cm_normalized_df}

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_prediction_callback_args = parser.add_argument_group(
            title='trans_svnet_sptial_module specific args options')
        return parser
