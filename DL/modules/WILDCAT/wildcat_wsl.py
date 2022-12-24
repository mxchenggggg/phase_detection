from modules.mastoid.mastoid_module_base import MastoidModuleBase
from modules.mastoid.mastoid_predictions_callback_base import MastoidPredictionsCallbackBase
from typing import List, Dict
import torch
import pandas as pd
import numpy as np
import os
import pickle
import pytorch_lightning as pl


class WildcatWSLPredClbk(MastoidPredictionsCallbackBase):
    def _split_predictions_outputs_by_videos(
            self, module: MastoidModuleBase, pred_outputs: List) -> Dict:
        # merge all batch results
        all_preds = []
        all_targets = []
        all_maps = []
        for outputs in pred_outputs:
            all_preds.append(outputs["preds"])
            all_targets.append(outputs["targets"])
            all_maps.append(outputs["maps"])
        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)
        all_maps = torch.vstack(all_maps)

        # split outputs by videos
        vid_indexes = module.datamodule.vid_idxes["pred"]
        df = module.datamodule.metadata["pred"]
        outputs_by_videos = {}
        for video_idx in vid_indexes:
            # row indexes in df corresponding to the video
            idxes = df.index[df[module.datamodule.video_index_col] == video_idx]

            preds = all_preds[idxes]
            targets = all_targets[idxes]
            maps = all_maps[idxes, :]
            img_paths = df.loc[idxes, module.datamodule.path_col]
            print(type(img_paths))

            outputs_by_videos[video_idx] = {
                "preds": preds, "targets": targets,
                "maps": maps, "img_paths": img_paths}

        return outputs_by_videos

    def _on_prediction_end_operations(
            self, trainer: pl.Trainer, module: MastoidModuleBase) -> None:
        outputs_by_videos = super()._on_prediction_end_operations(trainer, module)

        # output metadata file
        metadata = pd.DataFrame(columns=["map_path", "img_path", "video_index"])

        print("saving maps...")
        min_map_pixel_value = float('inf')
        max_map_pixel_value = -float('inf')
        for video_idx, outputs in outputs_by_videos.items():
            for i in range(len(outputs["img_paths"])):
                maps = outputs["maps"][i].cpu().numpy().astype(
                    np.float64)
                min_map_pixel_value = min(min_map_pixel_value, np.min(maps))
                max_map_pixel_value = max(max_map_pixel_value, np.max(maps))
                data = {
                    "maps": maps,
                    "target": outputs["targets"][i].cpu().numpy(),
                    "pred": outputs["preds"][i].cpu().numpy().astype(
                        np.float64)}
                img_path = outputs["img_paths"].iloc[i]
                img_file_name = os.path.splitext(os.path.basename(img_path))[0]
                map_path = os.path.join(
                    self.hprms.pred_output_path,
                    f'{img_file_name}_maps.pkl')
                with open(map_path, "wb") as f:
                    pickle.dump(data, f)

                row = {
                    "map_path": map_path, "img_path": img_path,
                    "video_index": video_idx}
                metadata = metadata.append(row, ignore_index=True)

        # save metadata file
        metadata_file_name = "WILDCAT_WSL_maps_metadata"
        metadata.to_csv(
            os.path.join(
                self.hprms.pred_output_path, metadata_file_name + ".csv"),
            index=False)
        metadata.to_pickle(
            os.path.join(
                self.hprms.pred_output_path, metadata_file_name + ".pkl"))

        maps_normalize = "WILDCAT_WSL_maps_normalize"
        with open(os.path.join(self.hprms.pred_output_path, maps_normalize + ".txt"), "w") as f:
            f.write(f"{min_map_pixel_value} {max_map_pixel_value}")
