import unittest
from unittest.mock import MagicMock, patch

import torch
import networkx as nx

_esm_patcher = patch("esmc_protein_function.model.ESMC")
_tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
_esm_patcher.start()
_tok_patcher.start()

from esmc_protein_function.model import ESMCProteinFunction, MultiLabelClassifier


def tearDownModule():
    _esm_patcher.stop()
    _tok_patcher.stop()


def _mock_encoder(
    embedding_dimensions: int = 8,
    num_layers: int = 6,
    batch_size: int = 2,
    seq_len: int = 10,
) -> MagicMock:
    encoder = MagicMock()
    encoder.blocks = [MagicMock() for _ in range(num_layers)]
    encoder.transformer.blocks = [MagicMock() for _ in range(num_layers)]

    for block in encoder.transformer.blocks:
        param = MagicMock(spec_set=["requires_grad"])
        param.requires_grad = False
        block.parameters.return_value = [param]

    dummy_param = MagicMock(spec_set=["requires_grad"])
    dummy_param.requires_grad = True
    dummy_module = MagicMock()
    dummy_module.parameters.return_value = [dummy_param]
    encoder.modules.return_value = [dummy_module]

    encoder.forward.return_value.embeddings = torch.randn(
        batch_size, seq_len, embedding_dimensions
    )

    return encoder


class _BaseTest(unittest.TestCase):
    embedding_dimensions = 8
    num_heads = 2
    num_encoder_layers = 6

    def _make_model(
        self,
        mf_terms: dict[int, str] | None = None,
        bp_terms: dict[int, str] | None = None,
        cc_terms: dict[int, str] | None = None,
        encoder: MagicMock | None = None,
    ) -> ESMCProteinFunction:
        if mf_terms is None:
            mf_terms = {0: "GO:0001", 1: "GO:0002"}
        if bp_terms is None:
            bp_terms = {0: "GO:0003", 1: "GO:0004"}
        if cc_terms is None:
            cc_terms = {0: "GO:0005", 1: "GO:0006"}
        if encoder is None:
            encoder = _mock_encoder(
                embedding_dimensions=self.embedding_dimensions,
                num_layers=self.num_encoder_layers,
            )

        model = ESMCProteinFunction(
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_mf_pool_heads=self.num_heads,
            num_bp_pool_heads=self.num_heads,
            num_cc_pool_heads=self.num_heads,
            index_to_mf_term=mf_terms,
            index_to_bp_term=bp_terms,
            index_to_cc_term=cc_terms,
            use_flash_attention=False,
        )

        model.encoder = encoder

        return model


class TestESMCProteinFunctionInit(_BaseTest):
    def test_empty_index_to_mf_term_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self._make_model(mf_terms={})

    def test_empty_index_to_bp_term_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self._make_model(bp_terms={})

    def test_empty_index_to_cc_term_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self._make_model(cc_terms={})

    def test_valid_construction(self):
        model = self._make_model()
        self.assertIsInstance(model, ESMCProteinFunction)
        self.assertIsInstance(model.mf_head, MultiLabelClassifier)
        self.assertIsInstance(model.bp_head, MultiLabelClassifier)
        self.assertIsInstance(model.cc_head, MultiLabelClassifier)
        self.assertIsNone(model.graph)
        self.assertEqual(model.index_to_mf_term, {0: "GO:0001", 1: "GO:0002"})
        self.assertEqual(model.index_to_bp_term, {0: "GO:0003", 1: "GO:0004"})
        self.assertEqual(model.index_to_cc_term, {0: "GO:0005", 1: "GO:0006"})
        self.assertEqual(model.embedding_dimensions, 8)

    def test_num_encoder_layers_property(self):
        model = self._make_model()
        self.assertEqual(model.num_encoder_layers, 6)

    def test_num_params_property(self):
        model = self._make_model()
        self.assertGreater(model.num_params, 0)

    def test_freeze_base_disables_encoder_grad(self):
        model = self._make_model()
        model.freeze_base()

        for module in model.encoder.modules():
            for param in module.parameters():
                self.assertFalse(param.requires_grad)

    def test_unfreeze_last_k_encoder_layers(self):
        model = self._make_model()
        model.freeze_base()
        model.unfreeze_last_k_encoder_layers(1)

        last_block = model.encoder.transformer.blocks[-1]
        for param in last_block.parameters():
            self.assertTrue(param.requires_grad)

        for block in model.encoder.transformer.blocks[:-1]:
            for param in block.parameters():
                self.assertFalse(param.requires_grad)

    def test_unfreeze_last_k_zero_raises_assertion_error(self):
        model = self._make_model()
        with self.assertRaises(AssertionError):
            model.unfreeze_last_k_encoder_layers(0)

    def test_load_gene_ontology_stores_valid_dag(self):
        model = self._make_model()
        graph = nx.DiGraph()
        graph.add_edge("GO:0001", "GO:0002")
        model.load_gene_ontology(graph)
        self.assertIs(model.graph, graph)

    def test_load_gene_ontology_rejects_cyclic_graph(self):
        model = self._make_model()
        graph = nx.DiGraph()
        graph.add_edge("GO:0001", "GO:0002")
        graph.add_edge("GO:0002", "GO:0001")
        with self.assertRaises(AssertionError):
            model.load_gene_ontology(graph)
        self.assertIsNone(model.graph)


class TestESMCProteinFunctionHelpers(_BaseTest):
    def test_match_terms_filters_by_threshold(self):
        model = self._make_model()
        probs = torch.tensor([[0.9, 0.3], [0.4, 0.8]])
        terms = model._match_terms(probs, {0: "GO:0001", 1: "GO:0002"}, top_p=0.5)

        self.assertEqual(len(terms), 2)
        self.assertIn("GO:0001", terms[0])
        self.assertAlmostEqual(terms[0]["GO:0001"], 0.9)
        self.assertNotIn("GO:0002", terms[0])
        self.assertIn("GO:0002", terms[1])
        self.assertAlmostEqual(terms[1]["GO:0002"], 0.8)
        self.assertNotIn("GO:0001", terms[1])

    def test_match_terms_all_below_threshold(self):
        model = self._make_model()
        probs = torch.tensor([[0.1, 0.2]])
        terms = model._match_terms(probs, {0: "GO:0001", 1: "GO:0002"}, top_p=0.5)
        self.assertEqual(terms, [{}])

    def test_match_terms_all_above_threshold(self):
        model = self._make_model()
        probs = torch.tensor([[0.9, 0.8]])
        terms = model._match_terms(probs, {0: "GO:0001", 1: "GO:0002"}, top_p=0.5)
        self.assertEqual(set(terms[0].keys()), {"GO:0001", "GO:0002"})

    def test_match_terms_invalid_top_p_raises_assertion_error(self):
        model = self._make_model()
        probs = torch.tensor([[0.5]])
        with self.assertRaises(AssertionError):
            model._match_terms(probs, {0: "GO:0001"}, top_p=0.0)
        with self.assertRaises(AssertionError):
            model._match_terms(probs, {0: "GO:0001"}, top_p=1.5)

    def test_build_subgraphs_no_graph_raises_assertion_error(self):
        model = self._make_model()
        with self.assertRaises(AssertionError):
            model._build_subgraphs([{"GO:0001": 0.9}])

    def test_build_subgraphs_propagates_probabilities(self):
        model = self._make_model()
        graph = nx.DiGraph()
        graph.add_edge("GO:0001", "GO:0002")
        graph.add_edge("GO:0002", "GO:0003")
        model.load_gene_ontology(graph)

        subgraphs, probabilities = model._build_subgraphs([{"GO:0001": 0.9}])

        self.assertEqual(len(subgraphs), 1)
        self.assertEqual(len(probabilities), 1)
        self.assertIn("GO:0001", probabilities[0])
        self.assertIn("GO:0002", probabilities[0])
        self.assertIn("GO:0003", probabilities[0])
        self.assertEqual(probabilities[0]["GO:0001"], 0.9)
        self.assertEqual(probabilities[0]["GO:0002"], 0.9)
        self.assertEqual(probabilities[0]["GO:0003"], 0.9)

    def test_build_subgraphs_returns_subgraph(self):
        model = self._make_model()
        graph = nx.DiGraph()
        graph.add_edge("GO:0001", "GO:0002")
        model.load_gene_ontology(graph)

        subgraphs, probabilities = model._build_subgraphs([{"GO:0001": 0.9}])

        self.assertEqual(len(subgraphs), 1)
        self.assertIsInstance(subgraphs[0], nx.DiGraph)
        self.assertIn("GO:0001", subgraphs[0].nodes)
        self.assertIn("GO:0002", subgraphs[0].nodes)
        self.assertEqual(len(subgraphs[0].nodes), 2)


class TestESMCProteinFunctionForward(_BaseTest):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.encoder = _mock_encoder(
            embedding_dimensions=self.embedding_dimensions,
            num_layers=self.num_encoder_layers,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
        )
        self.model = self._make_model(encoder=self.encoder)
        self.x = torch.randint(0, 64, (self.batch_size, self.seq_len))

    def test_forward_all_returns_correct_shapes(self):
        z_mf, z_bp, z_cc = self.model.forward_all(self.x)
        self.assertEqual(z_mf.shape, (self.batch_size, 2))
        self.assertEqual(z_bp.shape, (self.batch_size, 2))
        self.assertEqual(z_cc.shape, (self.batch_size, 2))

    def test_forward_mf_returns_correct_shape(self):
        z = self.model.forward_mf(self.x)
        self.assertEqual(z.shape, (self.batch_size, 2))

    def test_forward_bp_returns_correct_shape(self):
        z = self.model.forward_bp(self.x)
        self.assertEqual(z.shape, (self.batch_size, 2))

    def test_forward_cc_returns_correct_shape(self):
        z = self.model.forward_cc(self.x)
        self.assertEqual(z.shape, (self.batch_size, 2))

    def test_predict_all_applies_sigmoid(self):
        z_mf, z_bp, z_cc = self.model.predict_all(self.x)
        self.assertTrue(torch.all(z_mf > 0.0).item())
        self.assertTrue(torch.all(z_mf < 1.0).item())
        self.assertTrue(torch.all(z_bp > 0.0).item())
        self.assertTrue(torch.all(z_bp < 1.0).item())
        self.assertTrue(torch.all(z_cc > 0.0).item())
        self.assertTrue(torch.all(z_cc < 1.0).item())

    def test_predict_mf_terms_returns_go_term_mappings(self):
        terms = self.model.predict_mf_terms(self.x, top_p=0.01)
        self.assertEqual(len(terms), self.batch_size)
        for sample_terms in terms:
            self.assertIsInstance(sample_terms, dict)
            for go_id, prob in sample_terms.items():
                self.assertIsInstance(go_id, str)
                self.assertIsInstance(prob, float)
                self.assertGreater(prob, 0.0)
                self.assertLess(prob, 1.0)

    def test_predict_all_terms_returns_three_aspects(self):
        mf_terms, bp_terms, cc_terms = self.model.predict_all_terms(self.x, top_p=0.01)
        self.assertEqual(len(mf_terms), self.batch_size)
        self.assertEqual(len(bp_terms), self.batch_size)
        self.assertEqual(len(cc_terms), self.batch_size)

    def test_predict_all_subgraphs_integration(self):
        graph = nx.DiGraph()
        graph.add_edge("GO:0001", "GO:0002")
        graph.add_edge("GO:0003", "GO:0004")
        graph.add_edge("GO:0005", "GO:0006")
        self.model.load_gene_ontology(graph)

        results = self.model.predict_all_subgraphs(self.x, top_p=0.01)

        self.assertEqual(len(results), 3)
        for subgraphs, probabilities in results:
            self.assertEqual(len(subgraphs), self.batch_size)
            self.assertEqual(len(probabilities), self.batch_size)
            self.assertIsInstance(subgraphs[0], nx.DiGraph)
            self.assertIsInstance(probabilities[0], dict)
