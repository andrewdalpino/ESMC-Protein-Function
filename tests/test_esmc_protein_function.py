import unittest

import torch

from networkx import DiGraph

from esmc_protein_function.model import ESMCProteinFunction

CONFIG = {
    "embedding_dimensions": 16,
    "num_heads": 2,
    "num_encoder_layers": 2,
    "num_mf_pool_heads": 2,
    "num_bp_pool_heads": 2,
    "num_cc_pool_heads": 2,
    "num_mf_layers": 1,
    "num_bp_layers": 1,
    "num_cc_layers": 1,
    "index_to_mf_term": {0: "GO:0000001", 1: "GO:0000002"},
    "index_to_bp_term": {0: "GO:0000003", 1: "GO:0000004"},
    "index_to_cc_term": {0: "GO:0000005", 1: "GO:0000006"},
    "use_flash_attention": False,
}


def make_model(**overrides) -> ESMCProteinFunction:
    kwargs = {**CONFIG, **overrides}
    return ESMCProteinFunction(**kwargs)


def make_model_with_graph() -> ESMCProteinFunction:
    model = make_model()
    graph = DiGraph()
    graph.add_edge("GO:0000001", "GO:0000002")
    graph.add_nodes_from(["GO:0000003", "GO:0000004", "GO:0000005", "GO:0000006"])
    model.load_gene_ontology(graph)
    return model


def _encode(model, sequence: str):
    tokens = model.tokenizer.encode(sequence)
    return torch.tensor(tokens).unsqueeze(0)


class TestESMCProteinFunctionInit(unittest.TestCase):
    def test_valid_parameters(self):
        model = make_model()
        self.assertIsInstance(model, ESMCProteinFunction)
        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.mf_head)
        self.assertIsNotNone(model.bp_head)
        self.assertIsNotNone(model.cc_head)
        self.assertIsNotNone(model.tokenizer)

    def test_empty_index_to_mf_term_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            make_model(index_to_mf_term={})

    def test_empty_index_to_bp_term_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            make_model(index_to_bp_term={})

    def test_empty_index_to_cc_term_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            make_model(index_to_cc_term={})

    def test_index_mappings_are_stored(self):
        model = make_model()
        self.assertEqual(model.index_to_mf_term, CONFIG["index_to_mf_term"])
        self.assertEqual(model.index_to_bp_term, CONFIG["index_to_bp_term"])
        self.assertEqual(model.index_to_cc_term, CONFIG["index_to_cc_term"])

    def test_graph_is_none_by_default(self):
        model = make_model()
        self.assertIsNone(model.graph)


class TestESMCProteinFunctionProperties(unittest.TestCase):
    def setUp(self):
        self.model = make_model()

    def test_num_encoder_layers(self):
        self.assertEqual(self.model.num_encoder_layers, CONFIG["num_encoder_layers"])

    def test_num_params(self):
        self.assertGreater(self.model.num_params, 0)

    def test_num_trainable_parameters(self):
        self.assertGreater(self.model.num_trainable_parameters, 0)

    def test_num_trainable_equals_num_params_initially(self):
        self.assertEqual(
            self.model.num_trainable_parameters,
            self.model.num_params,
        )


class TestESMCProteinFunctionTraining(unittest.TestCase):
    def setUp(self):
        self.model = make_model()

    def test_freeze_base_disables_gradients(self):
        self.model.freeze_base()
        for param in self.model.encoder.parameters():
            self.assertFalse(param.requires_grad)

    def test_freeze_base_preserves_head_gradients(self):
        self.model.freeze_base()
        for param in self.model.mf_head.parameters():
            self.assertTrue(param.requires_grad)
        for param in self.model.bp_head.parameters():
            self.assertTrue(param.requires_grad)
        for param in self.model.cc_head.parameters():
            self.assertTrue(param.requires_grad)

    def test_unfreeze_last_k_encoder_layers(self):
        self.model.freeze_base()
        self.model.unfreeze_last_k_encoder_layers(1)
        for param in self.model.encoder.transformer.blocks[-1].parameters():
            self.assertTrue(param.requires_grad)
        for param in self.model.encoder.transformer.blocks[0].parameters():
            self.assertFalse(param.requires_grad)

    def test_unfreeze_last_k_encoder_layers_with_zero_k_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.unfreeze_last_k_encoder_layers(0)

    def test_unfreeze_last_k_encoder_layers_with_negative_k_raises_assertion_error(
        self,
    ):
        with self.assertRaises(AssertionError):
            self.model.unfreeze_last_k_encoder_layers(-1)

    def test_quantize_weights(self):
        self.model.quantize_weights(group_size=16)
        self.assertIsInstance(self.model, ESMCProteinFunction)


class TestESMCProteinFunctionGO(unittest.TestCase):
    def setUp(self):
        self.model = make_model()

    def test_load_gene_ontology_valid(self):
        graph = DiGraph()
        graph.add_edge("GO:0000001", "GO:0000002")
        self.model.load_gene_ontology(graph)
        self.assertIsNotNone(self.model.graph)

    def test_load_gene_ontology_with_cyclic_graph_raises_assertion_error(self):
        graph = DiGraph()
        graph.add_edge("GO:0000001", "GO:0000002")
        graph.add_edge("GO:0000002", "GO:0000001")
        with self.assertRaises(AssertionError):
            self.model.load_gene_ontology(graph)

    def test_load_gene_ontology_stores_graph(self):
        graph = DiGraph()
        graph.add_node("GO:0000001")
        self.model.load_gene_ontology(graph)
        self.assertIsNotNone(self.model.graph)
        self.assertIn("GO:0000001", self.model.graph.nodes)


class TestESMCProteinFunctionForward(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.tokens = _encode(self.model, "MKTVRQERLKSI")

    def test_forward_mf_returns_correct_shape(self):
        out = self.model.forward_mf(self.tokens)
        self.assertEqual(out.shape, (1, 2))

    def test_forward_bp_returns_correct_shape(self):
        out = self.model.forward_bp(self.tokens)
        self.assertEqual(out.shape, (1, 2))

    def test_forward_cc_returns_correct_shape(self):
        out = self.model.forward_cc(self.tokens)
        self.assertEqual(out.shape, (1, 2))

    def test_forward_all_returns_tuple_of_tensors(self):
        result = self.model.forward_all(self.tokens)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_forward_all_returns_correct_shapes(self):
        z_mf, z_bp, z_cc = self.model.forward_all(self.tokens)
        self.assertEqual(z_mf.shape, (1, 2))
        self.assertEqual(z_bp.shape, (1, 2))
        self.assertEqual(z_cc.shape, (1, 2))

    def test_forward_mf_output_dtype(self):
        out = self.model.forward_mf(self.tokens)
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_is_not_inplace(self):
        x = self.tokens.clone()
        _ = self.model.forward_mf(x)
        self.assertTrue(torch.equal(x, self.tokens))


class TestESMCProteinFunctionPredict(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.tokens = _encode(self.model, "MKTVRQERLKSI")

    def test_predict_mf_outputs_probabilities(self):
        out = self.model.predict_mf(self.tokens)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_predict_bp_outputs_probabilities(self):
        out = self.model.predict_bp(self.tokens)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_predict_cc_outputs_probabilities(self):
        out = self.model.predict_cc(self.tokens)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_predict_all_returns_tuple(self):
        result = self.model.predict_all(self.tokens)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_predict_mf_terms_returns_list_of_dicts(self):
        terms = self.model.predict_mf_terms(self.tokens, top_p=0.5)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(terms), 1)
        self.assertIsInstance(terms[0], dict)
        for go_id, prob in terms[0].items():
            self.assertIsInstance(go_id, str)
            self.assertIsInstance(prob, float)

    def test_predict_bp_terms_returns_list_of_dicts(self):
        terms = self.model.predict_bp_terms(self.tokens, top_p=0.5)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(terms), 1)
        self.assertIsInstance(terms[0], dict)

    def test_predict_cc_terms_returns_list_of_dicts(self):
        terms = self.model.predict_cc_terms(self.tokens, top_p=0.5)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(terms), 1)
        self.assertIsInstance(terms[0], dict)

    def test_predict_all_terms_returns_tuple(self):
        result = self.model.predict_all_terms(self.tokens, top_p=0.5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_lower_top_p_includes_more_terms(self):
        low_terms = self.model.predict_mf_terms(self.tokens, top_p=0.01)
        high_terms = self.model.predict_mf_terms(self.tokens, top_p=0.99)
        self.assertGreaterEqual(len(low_terms[0]), len(high_terms[0]))


class TestESMCProteinFunctionInternal(unittest.TestCase):
    def setUp(self):
        self.model = make_model()

    def test_match_terms_filters_by_top_p(self):
        probs = torch.tensor([[0.1, 0.9], [0.6, 0.4]])
        mapping = {0: "GO:001", 1: "GO:002"}
        terms = self.model._match_terms(probs, mapping, top_p=0.5)
        self.assertEqual(len(terms), 2)
        self.assertAlmostEqual(terms[0]["GO:002"], 0.9)
        self.assertAlmostEqual(terms[1]["GO:001"], 0.6)

    def test_match_terms_with_top_p_one_returns_empty(self):
        probs = torch.tensor([[0.1, 0.9]])
        mapping = {0: "GO:001", 1: "GO:002"}
        terms = self.model._match_terms(probs, mapping, top_p=1.0)
        self.assertEqual(len(terms[0]), 0)

    def test_match_terms_with_multiple_samples(self):
        probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        mapping = {0: "GO:001", 1: "GO:002"}
        terms = self.model._match_terms(probs, mapping, top_p=0.5)
        self.assertEqual(len(terms), 3)
        self.assertIn("GO:001", terms[0])
        self.assertIn("GO:002", terms[1])
        self.assertIn("GO:001", terms[2])

    def test_build_subgraphs_without_graph_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model._build_subgraphs([{"GO:001": 0.9}])

    def test_build_subgraphs_with_loaded_graph(self):
        graph = DiGraph()
        graph.add_edge("GO:0000001", "GO:0000002")
        self.model.load_gene_ontology(graph)
        terms = [{"GO:0000001": 0.9}]
        subgraphs, probabilities = self.model._build_subgraphs(terms)
        self.assertEqual(len(subgraphs), 1)
        self.assertEqual(len(probabilities), 1)
        self.assertIn("GO:0000001", subgraphs[0].nodes)
        self.assertIn("GO:0000002", subgraphs[0].nodes)
        self.assertAlmostEqual(probabilities[0]["GO:0000002"], 0.9)

    def test_build_subgraphs_propagates_max_probability(self):
        graph = DiGraph()
        graph.add_edge("GO:0000001", "GO:0000002")
        self.model.load_gene_ontology(graph)
        terms = [{"GO:0000001": 0.9, "GO:0000002": 0.3}]
        subgraphs, probabilities = self.model._build_subgraphs(terms)
        self.assertAlmostEqual(probabilities[0]["GO:0000002"], 0.9)

    def test_match_terms_with_top_p_out_of_range_raises_assertion_error(self):
        probs = torch.tensor([[0.5, 0.5]])
        mapping = {0: "GO:001", 1: "GO:002"}
        with self.assertRaises(AssertionError):
            self.model._match_terms(probs, mapping, top_p=0.0)
        with self.assertRaises(AssertionError):
            self.model._match_terms(probs, mapping, top_p=1.5)
        with self.assertRaises(AssertionError):
            self.model._match_terms(probs, mapping, top_p=-0.1)


class TestESMCProteinFunctionSubgraphs(unittest.TestCase):
    def setUp(self):
        self.model = make_model_with_graph()
        self.tokens = _encode(self.model, "MKTVRQERLKSI")

    def test_predict_mf_subgraphs_returns_correct_structure(self):
        subgraphs, terms = self.model.predict_mf_subgraphs(self.tokens, top_p=0.5)
        self.assertIsInstance(subgraphs, list)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(subgraphs), 1)
        self.assertEqual(len(terms), 1)
        self.assertIsInstance(subgraphs[0], DiGraph)
        self.assertIsInstance(terms[0], dict)

    def test_predict_bp_subgraphs_returns_correct_structure(self):
        subgraphs, terms = self.model.predict_bp_subgraphs(self.tokens, top_p=0.5)
        self.assertIsInstance(subgraphs, list)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(subgraphs), 1)
        self.assertEqual(len(terms), 1)

    def test_predict_cc_subgraphs_returns_correct_structure(self):
        subgraphs, terms = self.model.predict_cc_subgraphs(self.tokens, top_p=0.5)
        self.assertIsInstance(subgraphs, list)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(subgraphs), 1)
        self.assertEqual(len(terms), 1)

    def test_predict_all_subgraphs_returns_correct_structure(self):
        result = self.model.predict_all_subgraphs(self.tokens, top_p=0.5)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        for subgraphs, terms in result:
            self.assertIsInstance(subgraphs, list)
            self.assertIsInstance(terms, list)
            self.assertEqual(len(subgraphs), 1)
            self.assertEqual(len(terms), 1)

    def test_predict_mf_subgraphs_without_graph_raises_assertion_error(self):
        model = make_model()
        with self.assertRaises(AssertionError):
            model.predict_mf_subgraphs(self.tokens, top_p=0.5)

    def test_predict_all_subgraphs_with_multiple_batches(self):
        model = make_model_with_graph()
        tokens = _encode(model, "MKTVRQERLKSI")
        tokens = tokens.repeat(2, 1)
        subgraphs, terms = model.predict_mf_subgraphs(tokens, top_p=0.5)
        self.assertEqual(len(subgraphs), 2)
        self.assertEqual(len(terms), 2)
