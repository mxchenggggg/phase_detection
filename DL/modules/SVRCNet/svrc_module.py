from modules.mastoid.mastoid_module_base import MastoidModuleBase
from modules.mastoid.mastoid_predictions_callback_base import MastoidPredictionsCallbackBase
from typing import List, Dict
import torch
import pandas as pd
import numpy as np
import os
import pickle
import pytorch_lightning as pl


class SVRCNetModule(MastoidModuleBase):
    def _get_seq_last_one(self, outputs):
        sl = self.hprms.sequence_length

        outputs["preds"] = outputs["preds"][sl-1::sl]
        outputs["targets"] = outputs["targets"][sl-1::sl]
        return outputs

    def _forward_and_loss(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        outputs = self.forward(batch)

        # calculate loss
        outputs["loss"] = self.loss(outputs)

        return self._get_seq_last_one(outputs)

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        outputs = self._get_seq_last_one(outputs)
        self.predictions_outputs.append(outputs)
        return outputs


class SVRCNetClbk(MastoidPredictionsCallbackBase):
    def _split_predictions_outputs_by_videos(
            self, module: MastoidModuleBase, pred_outputs: List) -> Dict:
        # merge all batch results
        all_preds = []
        all_targets = []
        for outputs in pred_outputs:
            all_preds.append(outputs["preds"])
            all_targets.append(outputs["targets"])
        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)

        # split outputs by videos
        seq_start_idxes = module.datamodule.datasets["pred"].valid_seq_start_indexes
        df = module.datamodule.metadata["pred"]

        seq_vid_idxes = df.loc[seq_start_idxes,
                               module.datamodule.video_index_col].to_numpy().squeeze()

        vid_indexes = module.datamodule.vid_idxes["pred"]
        outputs_by_videos = {}
        for video_idx in vid_indexes:
            # row indexes in df corresponding to the video
            idxes = (seq_vid_idxes == video_idx)

            preds = all_preds[idxes]
            targets = all_targets[idxes]

            outputs_by_videos[video_idx] = {
                "preds": preds, "targets": targets}

        return outputs_by_videos
