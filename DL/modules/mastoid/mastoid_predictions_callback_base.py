from modules.mastoid.mastoid_module_base import MastoidModuleBase
from torchmetrics import Accuracy, Precision, Recall
from typing import Dict, Any, Optional, List
import pytorch_lightning as pl
import configargparse
import pandas as pd
import os
import pickle
import torch.nn.functional as F


class MastoidPredictionsCallbackBase(pl.Callback):
    """ Mastoid predictions callback base class

        The prediction callback will be setup in mastoid_module and called at the 
        end of prediction by pytorch lightnings automatically.

        Operations in _on_prediction_end_operations will be performed with the
        callback is triggered. By default, it calls _by_video_eval_and_log which
        calculate and log by-video evaluation results, including Accuracy, Precision, Recall
        for all and per-class. Evaluation resutls will be save as pkl and txt in
        predictions_output directory setup in the config file

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
        return self._by_video_eval_and_log(module, module.predictions_outputs)

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

    def _by_video_eval_and_log(self, module: MastoidModuleBase,
                               epoch_outputs: List) -> Dict:
        # split predictions output by videos
        outputs_by_videos = self._split_predictions_outputs_by_videos(
            module, epoch_outputs)

        # class names and number of classes
        class_names = module.hprms.class_names

        # evaluation results data frames
        df_all_videos = {}

        # calculate and log per-class metric results
        for vid_idx, outputs in outputs_by_videos.items():
            # iterate all videos
            df = pd.DataFrame(
                columns=class_names + ["all"],
                index=module.pred_metric_names)
            for metric_name in module.pred_metric_names:
                # iterate all metrics
                preds = F.softmax(outputs["preds"], dim=1)
                targets = outputs["targets"]

                # calculate result by class
                metric_by_class = getattr(
                    module, f'pred_{metric_name}_by_class')
                result_by_class = metric_by_class(preds, targets)
                metric_by_class.reset()
                for i, class_name in enumerate(class_names):
                    # iterate all classes
                    val = result_by_class[i].item()
                    df.loc[metric_name, class_name] = val

                # calculate result for all
                metric_all = getattr(module, f'pred_{metric_name}_all')
                result_all = metric_all(preds, targets)
                df.loc[metric_name, "all"] = result_all.item()
                metric_all.reset()

            df_all_videos[vid_idx] = df

        # save evaluation results

        # pickle file
        pred_eval_result_file_pkl = os.path.join(
            os.path.abspath(self.hprms.pred_output_path),
            "prediction_evaluation_results.pkl")
        with open(pred_eval_result_file_pkl, "wb") as f:
            pickle.dump(df_all_videos, f)

        # text file
        pred_eval_result_file_txt = os.path.join(
            os.path.abspath(self.hprms.pred_output_path),
            "prediction_evaluation_results.txt")
        with open(pred_eval_result_file_txt, "w") as f:
            for vid_idx, df in df_all_videos.items():
                df = df.astype(float)
                f.write(f'Video {vid_idx}\n')
                df.to_csv(f, sep='\t', decimal='.', float_format="%.3f")
                f.write('\n')

        return outputs_by_videos

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_prediction_callback_args = parser.add_argument_group(
            title='trans_svnet_sptial_module specific args options')
        return parser
