from trainers.mastoid.mastoid_trainer_base import MastoidTrainerBase
from pytorch_lightning import Trainer
from os import path
import os
import torch
import pandas as pd


class SptialFeatureExtractorTrainer(MastoidTrainerBase):
    def _predict(self) -> None:
        # make prediction
        trainer = Trainer(
            gpus=self.hprms.gpus, logger=self.loggers,
            resume_from_checkpoint=self.hprms.resume_from_checkpoint)
        # list of prediction for each batch
        predictions = trainer.predict(
            self.module, datamodule=self.datamodule)

        # prediction feature vector index
        index = 0

        # metadata for predicted features, useful when using features as inputs for other networks
        metadata = pd.DataFrame(columns=["path", "class", "video_index"])

        # save prediction vector as pytorch tensor file
        print("saving predictions vectors...")
        for batch_pred in predictions:
            for i in range(batch_pred.size(dim=0)):
                # 1. define output file name corresponding to input file
                input_file_path = self.datamodule.metadata["pred"].loc[index,
                                                                       self.datamodule.path_col]
                output_file_name = 'spatial_' + path.splitext(
                    path.basename(input_file_path))[0]

                # 2. save tensor file
                output_file_path = path.join(
                    self.hprms.output_path, output_file_name + '.pt')
                pred_tensor = batch_pred[i, :]
                torch.save(pred_tensor, output_file_path)

                # 3. load metadata associated with input file
                phase_label = self.datamodule.metadata["pred"].loc[index,
                                                                   self.datamodule.label_col]
                video_index = self.datamodule.metadata["pred"].loc[index,
                                                                   self.datamodule.video_index_col]

                # 4. add row to metadata for the output feature tensor
                row = {
                    "path": output_file_path, "class": phase_label,
                    "video_index": video_index}
                metadata = metadata.append(row, ignore_index=True)

                # 5. update index
                index += 1

        # save metadata file
        metadata_file_name = "TransSVNet_Spatial_Features_metadata"
        metadata.to_csv(
            path.join(self.hprms.output_path, metadata_file_name + ".csv"),
            index=False)
        metadata.to_pickle(
            path.join(self.hprms.output_path, metadata_file_name + ".pkl"))
