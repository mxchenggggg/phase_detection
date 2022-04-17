from trainers.mastoid.mastoid_trainer_base import MastoidTrainerBase
from pytorch_lightning import Trainer
from os import path
import os
import torch
import pandas as pd
import numpy as np
import pickle


class SptialFeatureExtractorTrainer(MastoidTrainerBase):
    def _predict(self) -> None:
        # make prediction
        trainer = Trainer(gpus=self.hprms.gpus, logger=self.loggers,)

        # list of prediction for each batch
        batch_predictions = trainer.predict(
            ckpt_path=self.hprms.resume_from_checkpoint,
            datamodule=self.datamodule, model=self.module)

        # all predictions
        all_predictions = []
        all_labels = []
        for preds, ls in batch_predictions:
            all_predictions.append(preds)
            all_labels.append(ls)
        all_predictions = torch.vstack(all_predictions)
        all_labels = torch.cat(all_labels)

        # outptu metadata file
        metadata = pd.DataFrame(columns=["path", "video_index"])

        video_idxes = self.datamodule.vid_idxes["pred"]
        df = self.datamodule.metadata["pred"]
        print("saving prediction features...")
        for video_idx in video_idxes:
            # load features and labels for current video
            idxes = df.index[df[self.datamodule.video_index_col] == video_idx]

            spatial_features = all_predictions[idxes, :].numpy().astype(
                np.float64)
            labels = all_labels[idxes].numpy()

            labels_from_df = df.loc[idxes, self.datamodule.label_col].values
            assert np.array_equal(
                labels, labels_from_df), "wrong labels in predictions!"

            # save features and labels
            data = {"spatial_features": spatial_features,
                    "labels": labels}
            output_file_path = path.join(
                self.hprms.output_path,
                f'V{video_idx:02d}_spatial_features.pkl')
            with open(output_file_path, "wb") as f:
                pickle.dump(data, f)

            row = {"path": output_file_path, "video_index": video_idx}
            metadata = metadata.append(row, ignore_index=True)

        # save metadata file
        metadata_file_name = "TransSVNet_Spatial_Features_metadata"
        metadata.to_csv(
            path.join(self.hprms.output_path, metadata_file_name + ".csv"),
            index=False)
        metadata.to_pickle(
            path.join(self.hprms.output_path, metadata_file_name + ".pkl"))
