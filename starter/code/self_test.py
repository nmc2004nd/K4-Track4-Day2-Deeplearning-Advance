"""Kiểm tra nhanh các phần dễ sai của pipeline."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from inference import aggregate_views, fuse_conv_bn
from losses import FocalLoss, mix_batch


class PipelineSelfTest(unittest.TestCase):
    def test_focal_gamma_zero_equals_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(7, 9)
        labels = torch.randint(0, 9, (7,))
        actual = FocalLoss(gamma=0.0)(logits, labels)
        expected = F.cross_entropy(logits, labels)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_cutmix_shape_and_effective_lambda(self):
        torch.manual_seed(0)
        np.random.seed(0)
        images = torch.randn(8, 3, 32, 32)
        labels = torch.arange(8)
        mixed, (labels_a, labels_b, lam) = mix_batch(images, labels, alpha=1.0, mode="cutmix")
        self.assertEqual(mixed.shape, images.shape)
        self.assertTrue(torch.equal(labels_a, labels))
        self.assertEqual(labels_b.shape, labels.shape)
        self.assertGreaterEqual(lam, 0.0)
        self.assertLessEqual(lam, 1.0)

    def test_conv_bn_fusion_preserves_output(self):
        torch.manual_seed(0)
        network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
        ).eval()
        inputs = torch.randn(4, 3, 16, 16)
        with torch.inference_mode():
            expected = network(inputs)
            actual = fuse_conv_bn(network)(inputs)
        self.assertLess((expected - actual).abs().max().item(), 1e-5)

    def test_probability_aggregation_is_normalized(self):
        logits = [np.zeros((3, 9), dtype=np.float32), np.ones((3, 9), dtype=np.float32)]
        probabilities = aggregate_views(logits, space="prob")
        self.assertEqual(probabilities.shape, (3, 9))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
