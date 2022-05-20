from datasets.mastoid.mastoid_dataset import MastoidDatasetBase
import torch
from typing import Any
import pickle
import numpy as np

class SVRCDataset(MastoidDatasetBase):
    def __getitem__(self, index: int) -> Any:
        start_index = self.valid_seq_start_indexes[index]
        seq_length = self.seq_length
        image_list = []
        label_list = []
        for i in range(start_index, start_index + seq_length):
            # load image change yml file!
            path = self.df.loc[i, self.path_col]
            image_list.append(torch.load(path))
            # load label
            label = self.df.loc[i, self.label_col]
            label_list.append(torch.tensor(label))
        return torch.stack(image_list), torch.stack(label_list)