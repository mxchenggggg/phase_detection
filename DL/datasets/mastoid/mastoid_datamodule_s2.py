from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from datasets.mastoid.mastoid_transform import MastoidTrasform
import pandas as pd
from pathlib import Path
from albumentations import Compose
from typing import Optional, Any
import configargparse
from sklearn.model_selection import train_test_split
import numpy as np
import torch


class MastoidDataSSModule(LightningDataModule):
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
        self.video_indexes = getattr(hparams, "train_video_indexes")
        self.downsampled_fps = getattr(
            hparams, "train_downsampled_fps")

        for attr in ["label_col", "path_col", "video_index_col"]:
            setattr(self, attr, getattr(hparams, f"{attr}_name"))

        self.transform = transform
        # column names in df
        self.label_col = hparams.label_col_name
        self.path_col = hparams.path_col_name
        self.video_idx_col = hparams.video_index_col_name

        self.meta_index = {}
        self.labels = {}

    def prepare_data(self) -> None:
        """ Load and split dataset metadata
        """
        # read metadata for all videos
        metafile_path = Path.joinpath(
            self.data_root, self.dataset_metadata_file_path)
        # self.metadata["all"] = pd.read_pickle(metafile_path)
        self.metadata = pd.read_csv(metafile_path)

        assert not self.metadata.isnull().values.any(
        ), "Dataframe contains nan Elements"
        self.metadata = self.metadata.reset_index(drop=True)
        # Downsample
        self.metadata_downsampled = self.__split_metadata_donwsampled()
        # Get the valid sequency
        self.meta_index["all"] = self._get_valid_seq_start_indexes()
        self.labels["all"] = self.__get_labels("all")
        # Train val split
        self.meta_index["train"], self.meta_index["val"], self.labels["train"], self.labels["val"] = train_test_split(
            self.meta_index["all"], self.labels["all"], test_size=0.3, random_state=0)

        self.meta_index["val"], self.meta_index["test"], self.labels["val"], self.labels["test"] = train_test_split(
            self.meta_index["val"], self.labels["val"], test_size=0.1, random_state=0)

        self.meta_index["pred"] = self.meta_index["all"].copy()
        self.labels["pred"] = self.labels["val"].copy()

    def setup(self, stage: Optional[str] = None) -> None:
        """ Set up datasets for traning, validation, testing and prediction
        """
        # Need to modify the list to be accessed by the dataset
        self.datasets = {}
        for split in ["train", "val", "test", "pred"]:
            self.datasets[split] = self.DatasetClass(
                self.hprms, self.metadata_downsampled,
                self.seq_len, self.meta_index[split],
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
        sampler = None
        if split == "train":
            class_count = np.unique(self.labels["train"], return_counts=True)[1]
            print(class_count)
            weight = 1. / class_count
            samples_weight = weight[self.labels["train"]]
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

            return DataLoader(
                dataset=self.datasets[split],
                batch_size=self.hprms.batch_size, sampler=sampler,
                num_workers=self.hprms.num_workers)

        return DataLoader(
            dataset=self.datasets[split],
            batch_size=self.hprms.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=self.hprms.num_workers)

    def __split_metadata_donwsampled(self) -> pd.DataFrame:
        """ Split metadata for all videos for training, validation, testing and prediction

        Args:
            split (str): name of the split(train/val/test)

        Returns:
            pd.DataFrame: metadata Dataframe for the split
        """

        if 0 < self.downsampled_fps < self.original_fps:
            factor = int(self.original_fps / self.downsampled_fps)
            downsampled_df = pd.DataFrame(
                columns=list(self.metadata.columns.values))
            for video_idx in self.video_indexes:
                video_frames = self.metadata.loc[self.metadata
                                                 [self.video_idx_col] ==
                                                 video_idx][:: factor]
                downsampled_df = pd.concat([downsampled_df, video_frames])
            df = downsampled_df
        df.reset_index(drop=True, inplace=True)
        return df

    def _get_valid_seq_start_indexes(self):
        # here action means seqeunce of adjacent frames with same labels
        action_seq_start_indexes = [0]
        for v_index in self.video_indexes:
            df_vid = self.metadata_downsampled[self.metadata_downsampled
                                               [self.video_idx_col] == v_index]
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
            if self.metadata_downsampled.loc[start, self.label_col] not in [0, 1, 2]:
                continue
            end = action_seq_start_indexes[i + 1]

            # ignore actions has number of frames less than sequence length
            if end - start <= self.seq_len:
                continue
            indexes = list(range(start, end - self.seq_len + 1))
            lables_count[self.metadata_downsampled.loc[start,
                                                       self.label_col]] += len(indexes)

            valid_seq_start_indexes += indexes
        print(lables_count)
        return valid_seq_start_indexes

    def __get_labels(self, split: str) -> list:
        valid_index_list = self.meta_index[split]
        label = []
        for start_index in valid_index_list:
            phase = self.metadata_downsampled.loc[start_index, self.label_col]
            label.append(phase)
        return label

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
