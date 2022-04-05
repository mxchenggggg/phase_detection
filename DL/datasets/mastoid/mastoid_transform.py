from albumentations import (
    Compose,
    Resize,
    Normalize,
    ShiftScaleRotate,
)
from albumentations.pytorch.transforms import ToTensorV2
import configargparse


class MastoidTrasform:
    """ The MastoidTrasform class manages data transforms for MastoidDataset
    """

    def __init__(self, hparams) -> None:
        """ MastoidTrasform Constructor
        """
        # Normalization
        self.normalize = Normalize(mean=hparams.norm_mean, std=hparams.norm_std)

        # Resize
        self.resize = Resize(height=hparams.input_height,
                             width=hparams.input_width)

        # Training augmentation
        self.train_aug = ShiftScaleRotate(shift_limit=hparams.shift_limit,
                                          scale_limit=hparams.scale_limit,
                                          rotate_limit=hparams.rotate_limit,
                                          border_mode=hparams.border_mode,
                                          value=hparams.value,
                                          p=hparams.p)

        # transforms for trainig, validation, testing and prediction
        self.transforms = {}
        if hparams.apply_training_aug:
            self.transforms["train"] = Compose(
                [self.resize, self.train_aug, self.normalize, ToTensorV2()])
        else:
            self.transforms["train"] = Compose(
                [self.resize, self.normalize, ToTensorV2()])
        self.transforms["val"] = Compose(
            [self.resize, self.normalize, ToTensorV2()])
        self.transforms["test"] = self.transforms["val"]
        self.transforms["pred"] = self.transforms["val"]

    def get_transform(self, split: str) -> Compose:
        """_summary_

        Args:
            split (str): name of the split(train/val/test)

        Returns:
            Compose: transform for the split
        """
        return self.transforms[split]

    def add_specific_args(parser: configargparse.ArgParser):
        mastoid_transform_args = parser.add_argument_group(
            title='mastoid_transform specific args options')

        # normalization
        mastoid_transform_args.add_argument(
            "--norm_mean", type=float, nargs='+', required=True)
        mastoid_transform_args.add_argument(
            "--norm_std", type=float, nargs='+', required=True)

        # training augmentation
        mastoid_transform_args.add_argument(
            "--apply_training_aug", type=bool, default=False)
        mastoid_transform_args.add_argument(
            "--shift_limit", type=float, default=0.1)
        mastoid_transform_args.add_argument(
            "--scale_limit", type=float, nargs='+', default=[-0.2, 0.5])
        mastoid_transform_args.add_argument(
            "--rotate_limit", type=int, default=15)
        mastoid_transform_args.add_argument(
            "--border_mode", type=int, default=0)
        mastoid_transform_args.add_argument(
            "--value", type=int, default=0)
        mastoid_transform_args.add_argument(
            "--p", type=float, default=0.7)
        return parser
