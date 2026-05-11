import unittest

import torch

from esmc_protein_function.model import AttentionPool


class TestAttentionPoolInit(unittest.TestCase):
    def test_valid_parameters(self):
        pool = AttentionPool(embedding_dimensions=64, num_heads=4)
        self.assertIsInstance(pool, AttentionPool)
        self.assertEqual(pool.linear1.in_features, 64)
        self.assertEqual(pool.linear1.out_features, 4)
        self.assertEqual(pool.linear2.in_features, 4 * 64)
        self.assertEqual(pool.linear2.out_features, 64)

    def test_zero_embedding_dimensions_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            AttentionPool(embedding_dimensions=0, num_heads=4)

    def test_negative_embedding_dimensions_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            AttentionPool(embedding_dimensions=-1, num_heads=4)

    def test_zero_num_heads_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            AttentionPool(embedding_dimensions=64, num_heads=0)

    def test_negative_num_heads_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            AttentionPool(embedding_dimensions=64, num_heads=-1)


class TestAttentionPoolForward(unittest.TestCase):
    def setUp(self):
        self.pool = AttentionPool(embedding_dimensions=64, num_heads=4)

    def test_forward_returns_correct_shape(self):
        x = torch.randn(2, 10, 64)
        out = self.pool.forward(x)
        self.assertEqual(out.shape, (2, 64))

    def test_forward_with_single_token(self):
        x = torch.randn(2, 1, 64)
        out = self.pool.forward(x)
        self.assertEqual(out.shape, (2, 64))

    def test_forward_with_single_sample(self):
        x = torch.randn(1, 10, 64)
        out = self.pool.forward(x)
        self.assertEqual(out.shape, (1, 64))

    def test_forward_with_different_embedding_dimensions(self):
        pool = AttentionPool(embedding_dimensions=128, num_heads=8)
        x = torch.randn(2, 10, 128)
        out = pool.forward(x)
        self.assertEqual(out.shape, (2, 128))

    def test_forward_output_dtype_matches_input(self):
        x = torch.randn(2, 10, 64)
        out = self.pool.forward(x)
        self.assertEqual(out.dtype, x.dtype)

    def test_forward_is_not_inplace(self):
        x = torch.randn(2, 10, 64)
        x_copy = x.clone()
        _ = self.pool.forward(x)
        self.assertTrue(torch.equal(x, x_copy))
