import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from albumentations import Compose
from typing import Tuple, List, Any, Optional


class MastoidDatasetBase(Dataset):
    """ Mastoidectomy Surgical Phase Segmentation Dataset 
        base class
    """

    def __init__(self, df: pd.DataFrame, seq_length: int,
                 video_indexes: List[int],
                 transform: Optional[Compose] = None,
                 label_col: Optional[str] = "classs",
                 path_col: Optional[str] = "path",
                 video_idx_col: Optional[str] = "video_idx") -> None:
        """ MastoidDatasetBase Constroctor  

        Args:
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
            label_col (Optional[str], optional): label column name in df. Defaults to "class".
            path_col (Optional[str], optional): data file column name in df. Defaults to "path".
            video_idx_col (Optional[str], optional): video index column name in df. Defaults to "video_idx".
        """

        super().__init__()
        self.df = df
        self.seq_length = seq_length

        self.transform = transform

        # column names in df
        self.label_col = label_col
        self.path_col = path_col
        self.video_idx_col = video_idx_col

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
        """get data point at givin index. Derived class must implement this method

        Args:
            index (int): _description_

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


class MastoidPerFrameRawImgDataset(MastoidDatasetBase):
    """ Per-frame raw image datset
    """

    def __init__(
            self, df: pd.DataFrame, video_indexes: List[int],
            transform: Compose) -> None:
        """ MastoidPerFrameRawImgDataset

        Args:
            df (pd.DataFrame):  DataFrame containing metadata for all videos.
            video_indexes (List[int]): list of video indexes.
            transform (Compose): image data transform
        """
        # seq_length = 1 for per-frame data
        super().__init__(df, 1, video_indexes, transform=transform)

    def load_img(self, index: int) -> Any:
        """ Load raw image at given index

        Args:
            index (int): index

        Returns:
            Any: image as torch tensor
        """
        img_index = self.valid_seq_start_indexes[index]
        img_path = self.df.loc[img_index, self.path_col]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img.type(torch.FloatTensor)

    def __getitem__(self, index):
        return self.load_img(index), self.load_label(index)
