import unittest

import torch

from esmc_protein_function.model import (
    GOTermClassifier,
    MLPClassifier,
    FeedForwardBlock,
)


class TestGOTermClassifierInit(unittest.TestCase):
    def test_valid_parameters(self):
        clf = GOTermClassifier(
            embedding_dimensions=64, num_heads=4, num_layers=2, num_classes=10
        )
        self.assertIsInstance(clf, GOTermClassifier)
        self.assertIsNotNone(clf.pool)
        self.assertIsNotNone(clf.mlp)

    def test_zero_embedding_dimensions_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            GOTermClassifier(
                embedding_dimensions=0, num_heads=4, num_layers=2, num_classes=10
            )

    def test_zero_num_heads_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            GOTermClassifier(
                embedding_dimensions=64, num_heads=0, num_layers=2, num_classes=10
            )

    def test_zero_num_layers_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            GOTermClassifier(
                embedding_dimensions=64, num_heads=4, num_layers=0, num_classes=10
            )

    def test_zero_num_classes_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            GOTermClassifier(
                embedding_dimensions=64, num_heads=4, num_layers=2, num_classes=0
            )

    def test_single_class(self):
        clf = GOTermClassifier(
            embedding_dimensions=64, num_heads=4, num_layers=1, num_classes=1
        )
        self.assertEqual(clf.mlp.out.out_features, 1)


class TestGOTermClassifierForward(unittest.TestCase):
    def setUp(self):
        self.clf = GOTermClassifier(
            embedding_dimensions=64, num_heads=4, num_layers=2, num_classes=10
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
        clf = GOTermClassifier(
            embedding_dimensions=64, num_heads=4, num_layers=1, num_classes=1
        )
        x = torch.randn(2, 10, 64)
        out = clf.forward(x)
        self.assertEqual(out.shape, (2, 1))

    def test_forward_output_dtype(self):
        x = torch.randn(2, 10, 64)
        out = self.clf.forward(x)
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_is_not_inplace(self):
        x = torch.randn(2, 10, 64)
        x_copy = x.clone()
        _ = self.clf.forward(x)
        self.assertTrue(torch.equal(x, x_copy))


class TestMLPClassifierInit(unittest.TestCase):
    def test_valid_parameters(self):
        mlp = MLPClassifier(embedding_dimensions=64, num_layers=2, num_classes=10)
        self.assertIsInstance(mlp, MLPClassifier)
        self.assertEqual(len(mlp.layers), 2)
        self.assertEqual(mlp.out.out_features, 10)

    def test_zero_embedding_dimensions_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            MLPClassifier(embedding_dimensions=0, num_layers=2, num_classes=10)

    def test_zero_num_layers_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            MLPClassifier(embedding_dimensions=64, num_layers=0, num_classes=10)

    def test_zero_num_classes_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            MLPClassifier(embedding_dimensions=64, num_layers=2, num_classes=0)

    def test_single_layer(self):
        mlp = MLPClassifier(embedding_dimensions=64, num_layers=1, num_classes=10)
        self.assertEqual(len(mlp.layers), 1)

    def test_single_class(self):
        mlp = MLPClassifier(embedding_dimensions=64, num_layers=2, num_classes=1)
        self.assertEqual(mlp.out.out_features, 1)


class TestMLPClassifierForward(unittest.TestCase):
    def setUp(self):
        self.mlp = MLPClassifier(embedding_dimensions=64, num_layers=2, num_classes=10)

    def test_forward_returns_correct_shape(self):
        x = torch.randn(2, 64)
        out = self.mlp.forward(x)
        self.assertEqual(out.shape, (2, 10))

    def test_forward_with_single_sample(self):
        x = torch.randn(1, 64)
        out = self.mlp.forward(x)
        self.assertEqual(out.shape, (1, 10))

    def test_forward_with_single_class(self):
        mlp = MLPClassifier(embedding_dimensions=64, num_layers=1, num_classes=1)
        x = torch.randn(2, 64)
        out = mlp.forward(x)
        self.assertEqual(out.shape, (2, 1))

    def test_forward_output_dtype(self):
        x = torch.randn(2, 64)
        out = self.mlp.forward(x)
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_is_not_inplace(self):
        x = torch.randn(2, 64)
        x_copy = x.clone()
        _ = self.mlp.forward(x)
        self.assertTrue(torch.equal(x, x_copy))

    def test_forward_with_deep_network(self):
        mlp = MLPClassifier(embedding_dimensions=64, num_layers=5, num_classes=10)
        x = torch.randn(2, 64)
        out = mlp.forward(x)
        self.assertEqual(out.shape, (2, 10))


class TestFeedForwardBlockInit(unittest.TestCase):
    def test_valid_parameters(self):
        block = FeedForwardBlock(embedding_dimensions=64)
        self.assertIsInstance(block, FeedForwardBlock)
        self.assertEqual(block.linear.in_features, 64)
        self.assertEqual(block.linear.out_features, 128)

    def test_zero_embedding_dimensions_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            FeedForwardBlock(embedding_dimensions=0)


class TestFeedForwardBlockForward(unittest.TestCase):
    def setUp(self):
        self.block = FeedForwardBlock(embedding_dimensions=64)

    def test_forward_returns_correct_shape(self):
        x = torch.randn(2, 64)
        out = self.block.forward(x)
        self.assertEqual(out.shape, (2, 64))

    def test_forward_with_single_sample(self):
        x = torch.randn(1, 64)
        out = self.block.forward(x)
        self.assertEqual(out.shape, (1, 64))

    def test_forward_output_dtype(self):
        x = torch.randn(2, 64)
        out = self.block.forward(x)
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_is_not_inplace(self):
        x = torch.randn(2, 64)
        x_copy = x.clone()
        _ = self.block.forward(x)
        self.assertTrue(torch.equal(x, x_copy))

    def test_forward_with_different_dimensions(self):
        block = FeedForwardBlock(embedding_dimensions=128)
        x = torch.randn(2, 128)
        out = block.forward(x)
        self.assertEqual(out.shape, (2, 128))
