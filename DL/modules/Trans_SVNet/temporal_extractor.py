from modules.mastoid.mastoid_module_base import MastoidModuleBase
from modules.mastoid.mastoid_predictions_callback_base import MastoidPredictionsCallbackBase
from modules.mastoid.mastoid_callbacks import MastoidPerVideoDatasetPredClbk
from typing import List, Dict
import torch
import pandas as pd
import numpy as np
import os
import pickle
import pytorch_lightning as pl


class TransSVNetTemporalExtractor(MastoidModuleBase):
    def loss(self, fwd_outputs):
        loss = 0.0
        targets = fwd_outputs["targets"].squeeze()
        for preds in fwd_outputs["stage_preds"]:
            preds = preds.squeeze()
            loss += self.ce_loss(preds, targets)
        return loss / float(len(fwd_outputs["stage_preds"]))


class TransSVNetTempExtClbk(MastoidPerVideoDatasetPredClbk):
    def _on_prediction_end_operations(
            self, trainer: pl.Trainer, module: MastoidModuleBase) -> None:
        outputs_by_videos = super()._on_prediction_end_operations(trainer, module)

        metadata = pd.DataFrame(columns=["path", "video_index"])

        print("saving temporal features...")
        for video_idx, outputs in outputs_by_videos.items():
            # save features
            data = {
                "spatial_features": outputs["spatial_features"].cpu().numpy().astype(np.float64),
                "temporal_features": outputs["preds"].cpu().numpy().astype(np.float64),
                "targets": outputs["targets"].cpu().numpy()}

            output_file_path = os.path.join(
                self.hprms.pred_output_path, f'temporal_V{video_idx:02}.pkl')
            with open(output_file_path, 'wb') as f:
                pickle.dump(data, f)

            row = {"path": output_file_path, "video_index": video_idx}
            metadata = metadata.append(row, ignore_index=True)

        # save metadata file
        metadata_file_name = "TransSVNet_Temporal_Features_metadata"
        metadata.to_csv(
            os.path.join(
                self.hprms.pred_output_path, metadata_file_name + ".csv"),
            index=False)
        metadata.to_pickle(
            os.path.join(
                self.hprms.pred_output_path, metadata_file_name + ".pkl"))
