from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from datasets.mastoid.mastoid_transform import MastoidTrasform
import pandas as pd
from pathlib import Path
from albumentations import Compose
from typing import Optional, Any
import configargparse


class MastoidDataModule(LightningDataModule):
    """ Pytorch Lightning Data Module for Mastoidectomy 
        Surgical Phase Segmentation Dataset 
    """

    def __init__(self, hparams, DatasetClass,
                 transform: Optional[Any] = None) -> None:
        """ MastoidDataModule constructor

        Args:
            hparams (_type_): hyperparameters and setting read from config file
            DatasetClass (_type_): Dataset class
            transform (Optional[Compose], optional): Data transfrom. Defaults to None.
        """

        super().__init__()
        self.hprms = hparams

        self.DatasetClass = DatasetClass

        self.data_root = Path(self.hprms.data_root)
        self.dataset_metadata_file_path = self.hprms.metadata_file

        # sequence length
        self.seq_len = self.hprms.sequence_length

        # fps after downsampled
        self.original_fps = hparams.original_fps
        self.downsampled_fps = {}
        # video indexes for each split
        self.vid_idxes = {}
        for split in ["train", "val", "test", "pred"]:
            self.vid_idxes[split] = getattr(hparams, f"{split}_video_indexes")
            self.downsampled_fps[split] = getattr(
                hparams, f"{split}_downsampled_fps")

        for attr in ["label_col", "path_col", "video_index_col"]:
            setattr(self, attr, getattr(hparams, f"{attr}_name"))

        self.transform = transform

        self.metadata = {}

    def prepare_data(self) -> None:
        """ Load and split dataset metadata
        """
        # read metadata for all videos
        metafile_path = Path.joinpath(
            self.data_root, self.dataset_metadata_file_path)
        # self.metadata["all"] = pd.read_pickle(metafile_path)
        self.metadata["all"] = pd.read_csv(metafile_path)

        assert not self.metadata["all"].isnull().values.any(
        ), "Dataframe contains nan Elements"
        self.metadata["all"] = self.metadata["all"].reset_index(drop=True)

        # split and downsample metadata
        for split in ["train", "val", "test", "pred"]:
            self.metadata[split] = self.__split_metadata_donwsampled(split)

    def setup(self, stage: Optional[str] = None) -> None:
        """ Set up datasets for traning, validation, testing and prediction
        """
        self.datasets = {}
        for split in ["train", "val", "test", "pred"]:
            self.datasets[split] = self.DatasetClass(
                self.hprms, self.metadata[split],
                self.seq_len, self.vid_idxes[split],
                transform=self.transform.get_transform(split))
            print(f"{split} dataset length: {self.datasets[split].__len__()}\n")

    def train_dataloader(self) -> DataLoader:
        return self.__get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self.__get_dataloader("test")

    def predict_dataloader(self):
        return self.__get_dataloader("pred")

    def __get_dataloader(self, split: str) -> DataLoader:
        shuffle = False
        if split == "train":
            shuffle = True
        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.hprms.batch_size, shuffle=shuffle,
            num_workers=self.hprms.num_workers)

    def __split_metadata_donwsampled(self, split: str) -> pd.DataFrame:
        """ Split metadata for all videos for training, validation, testing and prediction

        Args:
            split (str): name of the split(train/val/test)

        Returns:
            pd.DataFrame: metadata Dataframe for the split
        """
        indexes = self.metadata["all"][self.video_index_col].isin(
            self.vid_idxes[split])
        df = self.metadata["all"][indexes]

        if 0 < self.downsampled_fps[split] < self.original_fps:
            factor = int(self.original_fps / self.downsampled_fps[split])
            downsampled_df = pd.DataFrame(columns=list(df.columns.values))
            for video_idx in self.vid_idxes[split]:
                video_frames = df.loc[df[self.video_index_col]
                                      == video_idx][::factor]
                downsampled_df = pd.concat([downsampled_df, video_frames])
            df = downsampled_df
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_datamodule_args = parser.add_argument_group(
            title='mastoid_datamodule specific args options')

        # metadata file
        mastoid_datamodule_args.add_argument(
            "--metadata_file", type=str, required=True)
        # column names in metadata file
        mastoid_datamodule_args.add_argument(
            "--label_col_name", type=str, default="class")
        mastoid_datamodule_args.add_argument(
            "--path_col_name", type=str, default="path")
        mastoid_datamodule_args.add_argument(
            "--video_index_col_name", type=str, default="video_idx")

        # video indexes for training, validation and testing
        mastoid_datamodule_args.add_argument(
            "--train_video_indexes", type=int, nargs='+', required=True)
        mastoid_datamodule_args.add_argument(
            "--val_video_indexes", type=int, nargs='+', required=True)
        mastoid_datamodule_args.add_argument(
            "--test_video_indexes", type=int, nargs='+', required=True)
        mastoid_datamodule_args.add_argument(
            "--pred_video_indexes", type=int, nargs='+', default=[])

        # downsampleing data
        mastoid_datamodule_args.add_argument(
            "--original_fps", default=30, type=int)
        mastoid_datamodule_args.add_argument(
            "--train_downsampled_fps", default=1, type=int)
        mastoid_datamodule_args.add_argument(
            "--val_downsampled_fps", default=1, type=int)
        mastoid_datamodule_args.add_argument(
            "--test_downsampled_fps", default=1, type=int)
        mastoid_datamodule_args.add_argument(
            "--pred_downsampled_fps", default=1, type=int)

        # number of workers for dataloader
        mastoid_datamodule_args.add_argument(
            "--num_workers", type=int, default=8)
        return parser
