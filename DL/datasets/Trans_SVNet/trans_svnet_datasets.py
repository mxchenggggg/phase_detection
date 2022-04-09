from datasets.mastoid.mastoid_dataset import MastoidDatasetBase
import torch
from typing import Any
import pickle

class TransSVNetTemporalPerVideoDataset(MastoidDatasetBase):
    def __getitem__(self, index: int) -> Any:
        start_index = self.valid_seq_start_indexes[index]
        video_len = self.video_lengths[index]
        spatial_feature_list = []
        label_list = []
        for i in range(start_index, start_index+video_len):
            # load feature
            path = self.df.loc[i, self.path_col]
            spatial_feature_list.append(torch.load(path))
            # load label
            label = self.df.loc[i, self.label_col]
            label_list.append(torch.tensor(label))
        return torch.stack(spatial_feature_list), torch.stack(label_list)

class TransSVNetTransformerPerVideoDataset(MastoidDatasetBase):
    def __getitem__(self, index: int) -> Any:
        path = self.df.loc[index, self.path_col]
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["spatial_features"], data["temporal_features"], data["labels"]