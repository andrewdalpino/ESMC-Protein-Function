import unittest

import torch

from esmc_protein_function.model import MultiLabelClassifier


class TestMultiLabelClassifierInit(unittest.TestCase):
    def test_valid_parameters(self):
        clf = MultiLabelClassifier(embedding_dimensions=64, num_heads=4, num_classes=10)
        self.assertIsInstance(clf, MultiLabelClassifier)
        self.assertEqual(clf.linear1.in_features, 64)
        self.assertEqual(clf.linear1.out_features, 128)
        self.assertEqual(clf.linear2.in_features, 64)
        self.assertEqual(clf.linear2.out_features, 10)

    def test_zero_embedding_dimensions_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            MultiLabelClassifier(embedding_dimensions=0, num_heads=4, num_classes=10)

    def test_negative_embedding_dimensions_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            MultiLabelClassifier(embedding_dimensions=-1, num_heads=4, num_classes=10)

    def test_zero_num_classes_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            MultiLabelClassifier(embedding_dimensions=64, num_heads=4, num_classes=0)

    def test_negative_num_classes_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            MultiLabelClassifier(embedding_dimensions=64, num_heads=4, num_classes=-1)

    def test_single_class(self):
        clf = MultiLabelClassifier(embedding_dimensions=64, num_heads=4, num_classes=1)
        self.assertEqual(clf.linear2.out_features, 1)


class TestMultiLabelClassifierForward(unittest.TestCase):
    def setUp(self):
        self.clf = MultiLabelClassifier(
            embedding_dimensions=64, num_heads=4, num_classes=10
        )

    def test_forward_returns_correct_shape(self):
        x = torch.randn(2, 10, 64)
        out = self.clf.forward(x)
        self.assertEqual(out.shape, (2, 10))

    def test_forward_with_single_sample(self):
        x = torch.randn(1, 10, 64)
        out = self.clf.forward(x)
        self.assertEqual(out.shape, (1, 10))

    def test_forward_with_single_class(self):
        clf = MultiLabelClassifier(embedding_dimensions=64, num_heads=4, num_classes=1)
        x = torch.randn(2, 10, 64)
        out = clf.forward(x)
        self.assertEqual(out.shape, (2, 1))

    def test_forward_returns_float_tensor(self):
        x = torch.randn(2, 10, 64)
        out = self.clf.forward(x)
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_is_not_inplace(self):
        x = torch.randn(2, 10, 64)
        x_copy = x.clone()
        _ = self.clf.forward(x)
        self.assertTrue(torch.equal(x, x_copy))
