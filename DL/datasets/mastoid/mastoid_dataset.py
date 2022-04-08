import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from albumentations import Compose
from typing import Tuple, List, Any, Optional
import configargparse


class MastoidDatasetBase(Dataset):
    """ Mastoidectomy Surgical Phase Segmentation Dataset 
        base class
    """

    def __init__(self, hparams, df: pd.DataFrame, seq_length: int,
                 video_indexes: List[int],
                 transform: Optional[Compose] = None,) -> None:
        """ MastoidDatasetBase Constroctor  

        Args:
            hparams (_type_): hyperparameters and setting read from config file
            df (pd.DataFrame): DataFrame containing metadata for all videos.
                Required columns: 
                    1. data file path
                    2. surgical phase label
                    3. video index to which data file corresponds
            seq_length (int): sequence length.
                Each data point is a subsequence of data (i.e raw image, spatial/temporal feature)
                from same video. 
                Use seq_length = 1 for per-frame data.
                Use seq_length = -1 if each data point is a video. Lengths of videos 
                is stored in self.video_lengths.
            video_indexes (List[int]): list of video indexes.
            transform (Optional[Compose], optional): data transform. Defaults to None.
        """

        super().__init__()
        self.hprms = hparams

        self.df = df
        self.seq_length = seq_length

        self.transform = transform

        # column names in df
        self.label_col = hparams.label_col_name
        self.path_col = hparams.path_col_name
        self.video_idx_col = hparams.video_index_col_name

        # Row indexes of df s.t. the subsequence of seq_length starting from
        #   the row at the index are from the same video
        self.valid_seq_start_indexes = []
        if seq_length > 1:
            for v_index in video_indexes:
                # row indexes of data from video v_index
                row_indexes = self.df.index[self.df[self.video_idx_col]
                                            == v_index].tolist()
                self.valid_seq_start_indexes += row_indexes[:len(
                    row_indexes) - self.seq_length + 1]
        elif seq_length == 1:
            # per-frame data
            self.valid_seq_start_indexes = self.df.index
        elif seq_length == -1:
            # per-video data
            self.video_lengths = []
            for v_index in video_indexes:
                # row indexes of data from video v_index
                row_indexes = self.df.index[self.df[self.video_idx_col]
                                            == v_index].tolist()
                self.video_lengths.append(len(row_indexes))
                self.valid_seq_start_indexes.append(row_indexes[0])

    def __getitem__(self, index: int) -> Any:
        """ Get data point at givin index. Derived class must implement this method

        Args:
            index (int): index

        Raises:
            NotImplementedError: not implemented

        Returns:
            Any: data point
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """ Length of dataset. Same as len(self.df) for per-frame datset

        Returns:
            int: length
        """
        return len(self.valid_seq_start_indexes)

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_datset_args = parser.add_argument_group(
            title='mastoid_datset specific args options')
        mastoid_datset_args.add_argument(
            "--sequence_length", type=int, required=True)
        return parser


class MastoidPerFrameRawImgDataset(MastoidDatasetBase):
    """ Per-frame raw image datset
    """

    def load_input_file(self, index: int) -> torch.Tensor:
        """ Load raw image at given index

        Args:
            index (int): index

        Returns:
            Any: image as torch tensor
        """
        img_index = self.valid_seq_start_indexes[index]
        img_path = self.df.loc[img_index, self.path_col]
        img = np.array(Image.open(img_path))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img.type(torch.FloatTensor)

    def load_label(self, index: int) -> torch.Tensor:
        return torch.tensor(
            int(self.df.iloc[index, self.df.columns.get_loc(self.label_col)]))

    def __getitem__(self, index):
        return self.load_input_file(index), self.load_label(index)
