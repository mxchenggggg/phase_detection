from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets.mastoid.mastoid_transform import MastoidTrasform
import pandas as pd
from pathlib import Path
from albumentations import Compose
from typing import Optional
import configargparse


class MastoidDataModule(LightningDataModule):
    """ Pytorch Lightning Data Module for Mastoidectomy 
        Surgical Phase Segmentation Dataset 
    """

    def __init__(
            self, hparams, DatasetClass, transform: Optional[Compose] = None,
            label_col: Optional[str] = "class", path_col: Optional[str] = "path",
            video_idx_col: Optional[str] = "video_idx") -> None:
        """ MastoidDataModule constructor

        Args:
            hparams (_type_): hyperparameters and setting read from config file
            DatasetClass (_type_): Dataset class
            transform (Optional[Compose], optional): Data transfrom. Defaults to None.
            label_col (Optional[str], optional): label column name in df. Defaults to "class".
            path_col (Optional[str], optional): data file column name in df. Defaults to "path".
            video_idx_col (Optional[str], optional): video index column name in dataset metadata csv file. 
                                                     Defaults to "video_idx".
        """

        super().__init__()
        self.hprms = hparams

        self.DatasetClass = DatasetClass

        # fps after downsampled
        self.downsampled_fps = {}
        self.downsampled_fps["train"] = hparams.fps_sampling
        self.downsampled_fps["val"] = hparams.fps_sampling
        self.downsampled_fps["test"] = hparams.fps_sampling_test

        # sequence length
        self.seq_len = self.hprms.sequence_length

        # TODO: use config file
        self.data_root = Path(self.hprms.data_root)
        self.dataset_metadata_file_path = "mastoid_split_250px_30fps.pkl"

        self.vid_idxes = {}
        self.vid_idxes["train"] = [1, 3, 4, 5, 6, 7]
        self.vid_idxes["val"] = [8, 9, 10]
        self.vid_idxes["test"] = [12, 13, 14]

        self.label_col = label_col
        self.path_col = path_col
        self.video_idx_col = video_idx_col

        self.transform = transform

        self.metadata = {}

    def prepare_data(self) -> None:
        """ Load and split dataset metadata
        """
        # read metadata for all videos
        metafile_path = Path.joinpath(
            self.data_root, self.dataset_metadata_file_path)
        self.metadata["all"] = pd.read_pickle(metafile_path)

        # assert self.metadata["all"].isnull().values.any(
        # ), "Dataframe contains nan Elements"
        self.metadata["all"] = self.metadata["all"].reset_index()

        # split and downsample metadata
        for split in ["train", "val", "test"]:
            self.metadata[split] = self.__split_metadata_donwsampled(split)

    def setup(self, stage: Optional[str] = None) -> None:
        """ Set up datasets for traning, validation and testing
        """
        # TODO: pass parameters read from config file to MastoidTrasform
        self.datasets = {}
        for split in ["train", "val", "test"]:
            self.datasets[split] = self.DatasetClass(
                self.metadata[split],
                self.seq_len, self.vid_idxes[split],
                transform=self.transform, label_col=self.label_col,
                path_col=self.path_col, video_idx_col=self.video_idx_col)

    def train_dataloader(self) -> DataLoader:
        return self.__get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self.__get_dataloader("test")

    def __get_dataloader(self, split: str) -> DataLoader:
        shuffle = False
        if split == "train":
            shuffle = True
        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.hprms.batch_size, shuffle=shuffle,
            num_workers=self.hprms.num_workers)

    def __split_metadata_donwsampled(self, split: str) -> pd.DataFrame:
        """ Split metadata for all videos for training, validation and testing

        Args:
            split (str): name of the split(train/val/test)

        Returns:
            pd.DataFrame: metadata Dataframe for the split
        """
        indexes = self.metadata["all"][self.video_idx_col].isin(
            self.vid_idxes[split])
        df = self.metadata["all"][indexes]
        # TODO: magic number 30
        if 0 < self.downsampled_fps[split] < 30:
            factor = int(30 / self.downsampled_fps[split])
            df = df.iloc[::factor]
        return df

    @staticmethod
    def add_datamodule_specific_args(parser: configargparse.ArgParser):
        mastoid_datamodule = parser.add_argument_group(
            title='mastoid_datamodule specific args options')
        return parser
