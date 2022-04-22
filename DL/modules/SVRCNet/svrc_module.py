from modules.mastoid.mastoid_module_base import MastoidModuleBase
from modules.mastoid.mastoid_predictions_callback_base import MastoidPredictionsCallbackBase
from typing import List, Dict
import torch
import pandas as pd
import numpy as np
import os
import pickle
import pytorch_lightning as pl


class SVRCNetModule(MastoidPredictionsCallbackBase):
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
        vid_indexes = module.datamodule.vid_idxes["pred"]
        df = module.datamodule.metadata["pred"]
        outputs_by_videos = {}
        for video_idx in vid_indexes:
            # row indexes in df corresponding to the video
            idxes = df.index[df[module.datamodule.video_index_col] == video_idx]

            preds = all_preds[idxes]
            targets = all_targets[idxes]

            outputs_by_videos[video_idx] = {
                "preds": preds, "targets": targets}

        return outputs_by_videos

    def _on_prediction_end_operations(
            self, trainer: pl.Trainer, module: MastoidModuleBase) -> None:
        outputs_by_videos = super()._on_prediction_end_operations(trainer, module)

        # outptu metadata file
        metadata = pd.DataFrame(columns=["path", "video_index"])

        print("saving spatial features...")
        for video_idx, outputs in outputs_by_videos.items():
            # save spatial features and targets
            data = {
                "spatial_features":
                outputs["spatial_features"].cpu().numpy().astype(
                np.float64),
                "targets": outputs["targets"].cpu().numpy()}
            output_file_path = os.path.join(
                self.hprms.pred_output_path,
                f'V{video_idx:02d}_spatial_features.pkl')
            with open(output_file_path, "wb") as f:
                pickle.dump(data, f)

            # add row in metadata file
            row = {"path": output_file_path, "video_index": video_idx}
            metadata = metadata.append(row, ignore_index=True)

        # save metadata file
        metadata_file_name = "SVRC_metadata"
        metadata.to_csv(
            os.path.join(
                self.hprms.pred_output_path, metadata_file_name + ".csv"),
            index=False)
        metadata.to_pickle(
            os.path.join(
                self.hprms.pred_output_path, metadata_file_name + ".pkl"))
