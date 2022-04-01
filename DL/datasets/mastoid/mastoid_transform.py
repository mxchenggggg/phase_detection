from albumentations import (
    Compose,
    Resize,
    Normalize,
    ShiftScaleRotate,
)
from albumentations.pytorch.transforms import ToTensorV2


class MastoidTrasform:
    """ The MastoidTrasform class manages data transforms for MastoidDataset
    """

    def __init__(self) -> None:
        """ MastoidTrasform Constructor
            TODO: use configuration file for all parameters
        """
        # Normalization
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        self.normalize = Normalize(mean=norm_mean, std=norm_std)

        # Training augmentation
        shift_limit = 0.1,
        scale_limit = (-0.2, 0.5),
        rotate_limit = 15,
        border_mode = 0,
        value = 0,
        p = 0.7
        self.train_aug = ShiftScaleRotate(shift_limit=shift_limit,
                                          scale_limit=scale_limit,
                                          rotate_limit=rotate_limit,
                                          border_mode=border_mode,
                                          value=value,
                                          p=p)

        # Resize
        input_height = 224
        input_width = 224
        self.resize = Resize(height=input_height, width=input_width)

    def train_transform(self) -> Compose:
        """ Get data transform for training

        Returns:
            Compose: transform
        """
        return Compose(
            [self.resize, self.train_aug, self.normalize, ToTensorV2()])

    def val_transform(self) -> Compose:
        """ Get data transform for validation

        Returns:
            Compose: transform
        """
        return Compose([self.resize, self.normalize, ToTensorV2()])

    def test_transform(self) -> Compose:
        """ Get data transform for testing

        Returns:
            Compose: transform
        """
        return self.val_transform()
