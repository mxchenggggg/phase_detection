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
                 transform: Optional[Compose] = None) -> None:
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
        self.video_indexes = video_indexes

        self.transform = transform

        # column names in df
        self.label_col = hparams.label_col_name
        self.path_col = hparams.path_col_name
        self.video_idx_col = hparams.video_index_col_name

        # Row indexes of df s.t. the subsequence of seq_length starting from
        #   the row at the index are from the same video
        self.valid_seq_start_indexes = self._get_valid_seq_start_indexes()

    def _get_valid_seq_start_indexes(self):
        valid_seq_start_indexes = []
        if self.seq_length > 1:
            for v_index in self.video_indexes:
                # row indexes of data from video v_index
                row_indexes = self.df.index[self.df[self.video_idx_col]
                                            == v_index].tolist()
                valid_seq_start_indexes += row_indexes[:len(
                    row_indexes) - self.seq_length + 1]
        elif self.seq_length == 1:
            # per-frame data
            valid_seq_start_indexes = self.df.index
        elif self.seq_length == -1:
            # per-video data
            self.video_lengths = []
            for v_index in self.video_indexes:
                # row indexes of data from video v_index
                row_indexes = self.df.index[self.df[self.video_idx_col]
                                            == v_index].tolist()
                self.video_lengths.append(len(row_indexes))
                valid_seq_start_indexes.append(row_indexes[0])
        return valid_seq_start_indexes

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
        mastoid_dataset_args = parser.add_argument_group(
            title='mastoid_datset specific args options')
        mastoid_dataset_args.add_argument(
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
        label_index = self.valid_seq_start_indexes[index]
        label = int(self.df.loc[label_index, self.label_col])
        return torch.tensor(label)

    def __getitem__(self, index):
        return self.load_input_file(index), self.load_label(index)


class MastoidActionSeqDataset(MastoidDatasetBase):
    def _get_valid_seq_start_indexes(self):
        # here action means seqeunce of adjacent frames with same labels
        action_seq_start_indexes = [0]
        for v_index in self.video_indexes:
            df_vid = self.df[self.df[self.video_idx_col] == v_index]
            label_col_vid = getattr(df_vid, self.label_col)
            adj_check = (label_col_vid != label_col_vid.shift()).cumsum()
            vid_action_seq_start_indexes = df_vid.groupby(
                [self.label_col, adj_check],
                as_index=False, sort=False)[
                self.label_col].count().cumsum().to_numpy().squeeze() + action_seq_start_indexes[-1]
            action_seq_start_indexes += vid_action_seq_start_indexes.tolist()

        valid_seq_start_indexes = []
        lables_count = [0, 0, 0]
        for i in range(len(action_seq_start_indexes) - 1):
            start = action_seq_start_indexes[i]
            # TODO: only consider expose antrum and facial recess for now
            if self.df.loc[start, self.label_col] not in [0, 1, 2]:
                continue
            end = action_seq_start_indexes[i + 1]

            # ignore actions has number of frames less than sequence length
            if end - start <= self.seq_length:
                continue
            indexes = list(range(start, end - self.seq_length + 1))
            lables_count[self.df.loc[start, self.label_col]] += len(indexes)

            valid_seq_start_indexes += indexes
        print(lables_count)

        return valid_seq_start_indexes

    def __getitem__(self, index: int) -> Any:
        start_index = self.valid_seq_start_indexes[index]
        seq_length = self.seq_length
        image_list = []
        label_list = []
        for i in range(start_index, start_index + seq_length):
            # load image
            path = self.df.loc[i, self.path_col]
            img = np.array(Image.open(path))
            if self.transform:
                img = self.transform(image=img)["image"]
            img = img.type(torch.FloatTensor)
            image_list.append(img)
            # load label
            label = self.df.loc[i, self.label_col]
            label_list.append(torch.tensor(label))
        return torch.stack(image_list), torch.stack(label_list)


class MastoidActionSeqDataset_s2(Dataset):
    def __init__(self, hparams, df: pd.DataFrame, seq_length: int,
                 index_list: List[int],
                 transform: Optional[Compose] = None,) -> None:
        """ MastoidDataset for short sequence

        """

        super().__init__()
        self.hprms = hparams

        self.df = df
        self.seq_length = seq_length
        self.index_list = index_list

        self.transform = transform

        # column names in df
        self.label_col = hparams.label_col_name
        self.path_col = hparams.path_col_name
        self.video_idx_col = hparams.video_index_col_name

    def __len__(self) -> int:
        """ Length of dataset. Same as len(self.index_list) for short sequence datset

        Returns:
            int: length
        """
        return len(self.index_list)

    def __getitem__(self, index: int) -> Any:
        start_index = self.index_list[index]
        seq_length = self.seq_length
        image_list = []
        label_list = []
        for i in range(start_index, start_index + seq_length):
            # load image
            path = self.df.loc[i, self.path_col]
            img = np.array(Image.open(path))
            if self.transform:
                img = self.transform(image=img)["image"]
            img = img.type(torch.FloatTensor)
            image_list.append(img)
            # load label
            label = self.df.loc[i, self.label_col]
            label_list.append(torch.tensor(label))
        assert torch.all(torch.stack(
            label_list) == label_list[0]), f"The labels for the whole sequence should be the same, but got{label_list}"
        return torch.stack(image_list), torch.stack(label_list)

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_dataset_args = parser.add_argument_group(
            title='mastoid_datset specific args options')
        mastoid_dataset_args.add_argument(
            "--sequence_length", type=int, required=True)
        return parser
