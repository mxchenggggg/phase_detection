from models.WILDCAT.wildcat import *
import unittest
import argparse
import torch


class TestResNetWSL50(unittest.TestCase):
    def setUp(self) -> None:
        hparams = argparse.Namespace()
        hparams.kmax = 2
        hparams.kmin = 3
        hparams.alpha = 0.5
        hparams.num_maps = 4
        hparams.dense = False

        hparams.num_classes = 5
        hparams.batch_size = 16
        hparams.input_height = 225
        hparams.input_width = 400

        self.x = torch.rand(hparams.batch_size, 3,
                            hparams.input_height, hparams.input_width)
        self.targets = torch.rand(hparams.batch_size, 1)

        self.batch = (self.x, self.targets)

        self.hprms = hparams
        return super().setUp()

    def test_forward(self):
        model = ResNetWSL50(self.hprms)
        result = model.forward(self.batch)
        self.assertEqual(result['preds'].shape, torch.Size(
            [self.hprms.batch_size, self.hprms.num_classes]))
        self.assertEqual(result['targets'].shape, torch.Size(
            [self.hprms.batch_size, 1]))


if __name__ == '__main__':
    unittest.main()
