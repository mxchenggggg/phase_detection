from trainers.mastoid.mastoid_trainer_base import MastoidTrainerBase
from pytorch_lightning import Trainer
from os import path
import torch
import pandas as pd
import pickle


class TemporalFeatureExtractorTrainer(MastoidTrainerBase):
    def _get_input_spatial_features_and_labels(self, index: int):
        dataset = self.datamodule.datasets["pred"]
        start_index = dataset.valid_seq_start_indexes[index]
        video_len = dataset.video_lengths[index]
        spatial_feature_list = []
        label_list = []
        for i in range(start_index, start_index+video_len):
            # load feature
            path = dataset.df.loc[i, dataset.path_col]
            spatial_feature_list.append(torch.load(path))
            # load label
            label = dataset.df.loc[i, dataset.label_col]
            label_list.append(torch.tensor(label))
        return torch.stack(spatial_feature_list), torch.stack(label_list)

    def _predict(self) -> None:
        # make prediction
        trainer = Trainer(
            gpus=self.hprms.gpus, logger=self.loggers,
            resume_from_checkpoint=self.hprms.resume_from_checkpoint)

        # list of prediction for each batch, each batch is a video
        predictions = trainer.predict(
            self.module, datamodule=self.datamodule)

        metadata = pd.DataFrame(columns=["path", "video_index"])

        print("saving predictions features...")
        for idx in range(len(predictions)):
            # 1. temporal_features [video_length, out_features]
            temporal_features = predictions[idx]

            # 2. spatial_features: [video_length, 2048], labels: [video_length, 1]
            spatial_features, labels = self._get_input_spatial_features_and_labels(
                idx)

            # 3. save features
            data = {"spatial_features": spatial_features,
                    "temporal_features": temporal_features,  "labels": labels}
            video_index = self.datamodule.vid_idxes["pred"][idx]
            output_file_path = path.join(
                self.hprms.output_path, f'temporal_V{video_index:03}.pkl')
            with open(output_file_path, 'wb') as f:
                pickle.dump(data, f)

            # 4. add row to metadata

            row = {"path": output_file_path, "video_index": video_index}
            metadata = metadata.append(row, ignore_index=True)

        # save metadata file
        metadata_file_name = "TransSVNet_Temporal_Features_metadata"
        metadata.to_csv(
            path.join(self.hprms.output_path, metadata_file_name + ".csv"),
            index=False)
        metadata.to_pickle(
            path.join(self.hprms.output_path, metadata_file_name + ".pkl"))
