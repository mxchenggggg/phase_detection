import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from albumentations import Compose
from typing import Tuple, List, Any, Optional
import configargparse


class ToolMapDataset(Dataset):
    def __init__(self, hparams, df: pd.DataFrame, seq_length: int,
                 video_indexes: List[int],
                 transform: Optional[Compose] = None) -> None:
        super().__init__()
        self.hprms = hparams

        self.df = df
        self.seq_length = seq_length
        self.video_indexes = video_indexes

        self.transform = transform

        # column names in df
        self.label_col = hparams.label_col_name
        self.path_col = hparams.path_col_name
        self.video_idx_col = hparams.video_index_col_name

    def __getitem__(self, index: int) -> Any:
        return self.load_input_file(index), self.load_label(index)

    def __len__(self) -> int:
        return len(self.df)

    def load_input_file(self, index: int) -> torch.Tensor:
        img_path = self.df.loc[index, self.path_col]
        img = np.array(Image.open(img_path))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img.type(torch.FloatTensor)

    def load_label(self, index: int) -> torch.Tensor:
        label = int(self.df.loc[index, self.label_col])
        return torch.tensor(label)

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_dataset_args = parser.add_argument_group(
            title='tool_map_datset specific args options')
        mastoid_dataset_args.add_argument(
            "--sequence_length", type=int, required=True)
        return parser
