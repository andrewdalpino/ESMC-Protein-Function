import unittest
import warnings

import torch

from networkx import DiGraph
from esmc_protein_function.model import ESMCProteinFunction


class TestESMCProteinFunctionInit(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "embedding_dimensions": 8,
            "num_heads": 2,
            "num_encoder_layers": 1,
            "num_pool_heads": 2,
            "indexToMfGoTerm": {0: "GO:0003674", 1: "GO:0003824"},
            "indexToBpGoTerm": {0: "GO:0008150", 1: "GO:0009987"},
            "indexToCcGoTerm": {0: "GO:0005575", 1: "GO:0005634"},
            "use_flash_attention": False,
        }

    def test_valid_parameters(self):
        model = ESMCProteinFunction(**self.kwargs)
        self.assertIsInstance(model, ESMCProteinFunction)
        self.assertEqual(model.embedding_dimensions, 8)
        self.assertEqual(len(model.encoder.transformer.blocks), 1)

    def test_zero_embedding_dimensions_raises_assertion_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(AssertionError):
                ESMCProteinFunction(**{**self.kwargs, "embedding_dimensions": 0})

    def test_negative_embedding_dimensions_raises_error(self):
        with self.assertRaises(RuntimeError):
            ESMCProteinFunction(**{**self.kwargs, "embedding_dimensions": -1})

    def test_empty_go_term_mapping_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            ESMCProteinFunction(**{**self.kwargs, "indexToMfGoTerm": {}})

    def test_num_params_property(self):
        model = ESMCProteinFunction(**self.kwargs)
        self.assertGreater(model.num_params, 0)

    def test_num_trainable_parameters_property(self):
        model = ESMCProteinFunction(**self.kwargs)
        self.assertEqual(model.num_params, model.num_trainable_parameters)

    def test_freeze_base(self):
        model = ESMCProteinFunction(**self.kwargs)
        model.freeze_base()
        for param in model.encoder.parameters():
            self.assertFalse(param.requires_grad)
        for head in [model.mf_head, model.bp_head, model.cc_head]:
            for param in head.parameters():
                self.assertTrue(param.requires_grad)

    def test_unfreeze_last_k_encoder_layers(self):
        model = ESMCProteinFunction(**self.kwargs)
        model.freeze_base()
        model.unfreeze_last_k_encoder_layers(k=1)
        self.assertTrue(any(p.requires_grad for p in model.encoder.parameters()))


class TestESMCProteinFunctionForward(unittest.TestCase):
    def setUp(self):
        self.model = ESMCProteinFunction(
            embedding_dimensions=8,
            num_heads=2,
            num_encoder_layers=1,
            num_pool_heads=2,
            indexToMfGoTerm={0: "GO:0003674", 1: "GO:0003824"},
            indexToBpGoTerm={0: "GO:0008150", 1: "GO:0009987"},
            indexToCcGoTerm={0: "GO:0005575", 1: "GO:0005634"},
            use_flash_attention=False,
        )
        self.model.eval()
        encoding = self.model.tokenizer("MKTAYIA", return_tensors="pt")
        self.x = encoding["input_ids"]

    def test_forward_mf_returns_correct_shape(self):
        out = self.model.forward_mf(self.x)
        self.assertEqual(out.shape, (1, 2))

    def test_forward_bp_returns_correct_shape(self):
        out = self.model.forward_bp(self.x)
        self.assertEqual(out.shape, (1, 2))

    def test_forward_cc_returns_correct_shape(self):
        out = self.model.forward_cc(self.x)
        self.assertEqual(out.shape, (1, 2))

    def test_forward_all_returns_tuple_of_three_tensors(self):
        mf, bp, cc = self.model.forward_all(self.x)
        self.assertIsInstance(mf, torch.Tensor)
        self.assertIsInstance(bp, torch.Tensor)
        self.assertIsInstance(cc, torch.Tensor)
        self.assertEqual(mf.shape, (1, 2))
        self.assertEqual(bp.shape, (1, 2))
        self.assertEqual(cc.shape, (1, 2))

    def test_forward_output_dtype(self):
        out = self.model.forward_mf(self.x)
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_is_not_inplace(self):
        x_copy = self.x.clone()
        _ = self.model.forward_mf(self.x)
        self.assertTrue(torch.equal(self.x, x_copy))

    def test_predict_mf_values_in_range(self):
        out = self.model.predict_mf(self.x)
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_predict_all_values_in_range(self):
        mf, bp, cc = self.model.predict_all(self.x)
        for out in [mf, bp, cc]:
            self.assertTrue(torch.all(out >= 0.0))
            self.assertTrue(torch.all(out <= 1.0))

    def test_predict_mf_terms_returns_list_of_dicts(self):
        terms = self.model.predict_mf_terms(self.x, top_p=0.5)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(terms), 1)
        self.assertIsInstance(terms[0], dict)

    def test_predict_mf_terms_high_threshold_returns_empty_dicts(self):
        terms = self.model.predict_mf_terms(self.x, top_p=1.0)
        self.assertEqual(len(terms), 1)
        self.assertEqual(len(terms[0]), 0)

    def test_predict_all_terms_returns_tuple(self):
        mf, bp, cc = self.model.predict_all_terms(self.x, top_p=0.5)
        self.assertEqual(len(mf), 1)
        self.assertEqual(len(bp), 1)
        self.assertEqual(len(cc), 1)
        self.assertIsInstance(mf[0], dict)
        self.assertIsInstance(bp[0], dict)
        self.assertIsInstance(cc[0], dict)


class TestESMCProteinFunctionGraph(unittest.TestCase):
    def setUp(self):
        self.model = ESMCProteinFunction(
            embedding_dimensions=8,
            num_heads=2,
            num_encoder_layers=1,
            num_pool_heads=2,
            indexToMfGoTerm={0: "GO:0003674", 1: "GO:0003824"},
            indexToBpGoTerm={0: "GO:0008150", 1: "GO:0009987"},
            indexToCcGoTerm={0: "GO:0005575", 1: "GO:0005634"},
            use_flash_attention=False,
        )

    def test_load_gene_ontology_valid_graph(self):
        graph = DiGraph()
        graph.add_edge("GO:0003674", "GO:0003824")
        self.model.load_gene_ontology(graph)
        self.assertIsNotNone(self.model.graph)

    def test_load_gene_ontology_invalid_graph_raises_assertion_error(self):
        graph = DiGraph()
        graph.add_edge("GO:0003674", "GO:0003824")
        graph.add_edge("GO:0003824", "GO:0003674")
        with self.assertRaises(AssertionError):
            self.model.load_gene_ontology(graph)
