from datasets.mastoid.mastoid_dataset import MastoidDatasetBase
import torch
from typing import Any
import pickle
import numpy as np


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


class TransSVNetTemporalPerVideoDatasetSimple(MastoidDatasetBase):
    def __getitem__(self, index: int) -> Any:
        path = self.df.loc[index, self.path_col]
        with open(path, "rb") as f:
            data = pickle.load(f)

        return torch.from_numpy(
            data["spatial_features"]).type(torch.FloatTensor), torch.from_numpy(
            data["targets"].astype(int)).type(torch.LongTensor)


class TransSVNetTransformerPerVideoDataset(MastoidDatasetBase):
    def __getitem__(self, index: int) -> Any:
        path = self.df.loc[index, self.path_col]
        with open(path, "rb") as f:
            data = pickle.load(f)
        return torch.from_numpy(
            data["spatial_features"]).type(
            torch.FloatTensor), torch.from_numpy(
            data["temporal_features"]).type(
            torch.FloatTensor), torch.from_numpy(
            data["targets"]).type(
            torch.LongTensor)
