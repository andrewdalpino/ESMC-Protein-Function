import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.nn import Identity, Parameter
from networkx import DiGraph

from esmc_protein_function.model import (
    ESMCProteinFunction,
    GOTermClassifier,
)

NUM_MF_CLASSES = 3
NUM_BP_CLASSES = 3
NUM_CC_CLASSES = 3
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2


def _make_model(encoder=None):
    if encoder is None:
        encoder = MagicMock()
        encoder.transformer.blocks = [MagicMock() for _ in range(NUM_LAYERS)]

    with (
        patch("esmc_protein_function.model.ESMC", return_value=encoder),
        patch("esmc_protein_function.model.EsmSequenceTokenizer"),
    ):
        model = ESMCProteinFunction(
            embedding_dimensions=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_mf_pool_heads=2,
            num_bp_pool_heads=2,
            num_cc_pool_heads=2,
            num_mf_layers=2,
            num_bp_layers=2,
            num_cc_layers=2,
            index_to_mf_term={i: f"GO:00000{i}" for i in range(NUM_MF_CLASSES)},
            index_to_bp_term={i: f"GO:00001{i}" for i in range(NUM_BP_CLASSES)},
            index_to_cc_term={i: f"GO:00002{i}" for i in range(NUM_CC_CLASSES)},
            use_flash_attention=False,
        )

    return model


class TestESMCProteinFunctionInit(unittest.TestCase):
    @patch("esmc_protein_function.model.ESMC")
    @patch("esmc_protein_function.model.EsmSequenceTokenizer")
    def test_valid_initialization(self, mock_tokenizer, mock_esmc):
        encoder = MagicMock()
        encoder.transformer.blocks = [MagicMock(), MagicMock()]
        mock_esmc.return_value = encoder

        model = ESMCProteinFunction(
            embedding_dimensions=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_mf_pool_heads=2,
            num_bp_pool_heads=2,
            num_cc_pool_heads=2,
            num_mf_layers=2,
            num_bp_layers=2,
            num_cc_layers=2,
            index_to_mf_term={0: "GO:000001"},
            index_to_bp_term={0: "GO:000002"},
            index_to_cc_term={0: "GO:000003"},
            use_flash_attention=False,
        )

        self.assertIsInstance(model, ESMCProteinFunction)
        self.assertIsNotNone(model.encoder)
        self.assertIsInstance(model.mf_head, GOTermClassifier)
        self.assertIsInstance(model.bp_head, GOTermClassifier)
        self.assertIsInstance(model.cc_head, GOTermClassifier)
        self.assertEqual(model.embedding_dimensions, EMBEDDING_DIM)
        self.assertIsNone(model.graph)
        self.assertIsNotNone(model.tokenizer)

    @patch("esmc_protein_function.model.ESMC")
    @patch("esmc_protein_function.model.EsmSequenceTokenizer")
    def test_encoder_sequence_head_replaced_with_identity(
        self, mock_tokenizer, mock_esmc
    ):
        encoder = MagicMock()
        encoder.transformer.blocks = [MagicMock(), MagicMock()]
        mock_esmc.return_value = encoder

        ESMCProteinFunction(
            embedding_dimensions=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_mf_pool_heads=2,
            num_bp_pool_heads=2,
            num_cc_pool_heads=2,
            num_mf_layers=2,
            num_bp_layers=2,
            num_cc_layers=2,
            index_to_mf_term={0: "GO:000001"},
            index_to_bp_term={0: "GO:000002"},
            index_to_cc_term={0: "GO:000003"},
            use_flash_attention=False,
        )

        self.assertIsInstance(encoder.sequence_head, Identity)

    @patch("esmc_protein_function.model.ESMC")
    @patch("esmc_protein_function.model.EsmSequenceTokenizer")
    def test_empty_index_to_mf_term_raises_assertion_error(
        self, mock_tokenizer, mock_esmc
    ):
        with self.assertRaises(AssertionError):
            ESMCProteinFunction(
                embedding_dimensions=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                num_encoder_layers=NUM_LAYERS,
                num_mf_pool_heads=2,
                num_bp_pool_heads=2,
                num_cc_pool_heads=2,
                num_mf_layers=2,
                num_bp_layers=2,
                num_cc_layers=2,
                index_to_mf_term={},
                index_to_bp_term={0: "GO:000002"},
                index_to_cc_term={0: "GO:000003"},
                use_flash_attention=False,
            )

    @patch("esmc_protein_function.model.ESMC")
    @patch("esmc_protein_function.model.EsmSequenceTokenizer")
    def test_empty_index_to_bp_term_raises_assertion_error(
        self, mock_tokenizer, mock_esmc
    ):
        with self.assertRaises(AssertionError):
            ESMCProteinFunction(
                embedding_dimensions=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                num_encoder_layers=NUM_LAYERS,
                num_mf_pool_heads=2,
                num_bp_pool_heads=2,
                num_cc_pool_heads=2,
                num_mf_layers=2,
                num_bp_layers=2,
                num_cc_layers=2,
                index_to_mf_term={0: "GO:000001"},
                index_to_bp_term={},
                index_to_cc_term={0: "GO:000003"},
                use_flash_attention=False,
            )

    @patch("esmc_protein_function.model.ESMC")
    @patch("esmc_protein_function.model.EsmSequenceTokenizer")
    def test_empty_index_to_cc_term_raises_assertion_error(
        self, mock_tokenizer, mock_esmc
    ):
        with self.assertRaises(AssertionError):
            ESMCProteinFunction(
                embedding_dimensions=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                num_encoder_layers=NUM_LAYERS,
                num_mf_pool_heads=2,
                num_bp_pool_heads=2,
                num_cc_pool_heads=2,
                num_mf_layers=2,
                num_bp_layers=2,
                num_cc_layers=2,
                index_to_mf_term={0: "GO:000001"},
                index_to_bp_term={0: "GO:000002"},
                index_to_cc_term={},
                use_flash_attention=False,
            )

    @patch("esmc_protein_function.model.ESMC")
    @patch("esmc_protein_function.model.EsmSequenceTokenizer")
    def test_string_keys_are_converted_to_integers(self, mock_tokenizer, mock_esmc):
        encoder = MagicMock()
        encoder.transformer.blocks = [MagicMock(), MagicMock()]
        mock_esmc.return_value = encoder

        model = ESMCProteinFunction(
            embedding_dimensions=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_mf_pool_heads=2,
            num_bp_pool_heads=2,
            num_cc_pool_heads=2,
            num_mf_layers=2,
            num_bp_layers=2,
            num_cc_layers=2,
            index_to_mf_term={"0": "GO:000001", "1": "GO:000002"},
            index_to_bp_term={"0": "GO:000003"},
            index_to_cc_term={"0": "GO:000004"},
            use_flash_attention=False,
        )

        self.assertEqual(model.index_to_mf_term, {0: "GO:000001", 1: "GO:000002"})
        self.assertEqual(model.index_to_bp_term, {0: "GO:000003"})
        self.assertEqual(model.index_to_cc_term, {0: "GO:000004"})


class TestESMCProteinFunctionProperties(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()

    def test_num_encoder_layers(self):
        self.assertEqual(self.model.num_encoder_layers, NUM_LAYERS)

    def test_num_params_returns_positive_integer(self):
        self.assertGreater(self.model.num_params, 0)

    def test_num_trainable_parameters_returns_positive_integer(self):
        self.assertGreater(self.model.num_trainable_parameters, 0)


class TestESMCProteinFunctionEncoderFreezing(unittest.TestCase):
    def setUp(self):
        encoder = MagicMock()
        encoder.transformer.blocks = [MagicMock() for _ in range(3)]
        self.model = _make_model(encoder)
        self.encoder = encoder

    def test_freeze_encoder_sets_all_params_requires_grad_false(self):
        param = Parameter(torch.randn(5))
        module = MagicMock(spec=["parameters"])
        module.parameters.return_value = [param]
        self.encoder.modules.return_value = [module]

        self.model.freeze_encoder()

        self.assertFalse(param.requires_grad)

    def test_unfreeze_last_k_encoder_layers_sets_requires_grad_true(self):
        for i, block in enumerate(self.encoder.transformer.blocks):
            block.parameters.return_value = [
                Parameter(torch.randn(5), requires_grad=(i < 2))
            ]

        self.model.unfreeze_last_k_encoder_layers(1)

        for block in self.encoder.transformer.blocks[-1:]:
            for param in block.parameters():
                self.assertTrue(param.requires_grad)

    def test_unfreeze_last_k_with_zero_k_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.unfreeze_last_k_encoder_layers(0)

    def test_unfreeze_last_k_with_negative_k_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.unfreeze_last_k_encoder_layers(-1)


class TestESMCProteinFunctionQuantization(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()

    @patch("esmc_protein_function.model.quantize_")
    def test_add_fake_quantized_tensors(self, mock_quantize):
        self.model.add_fake_quantized_tensors(group_size=32)

        mock_quantize.assert_called_once()

    @patch("esmc_protein_function.model.quantize_")
    def test_remove_fake_quantized_tensors(self, mock_quantize):
        self.model.remove_fake_quantized_tensors()

        mock_quantize.assert_called_once()

    @patch("esmc_protein_function.model.quantize_")
    def test_quantize_weights(self, mock_quantize):
        self.model.quantize_weights(group_size=32)

        mock_quantize.assert_called_once()


class TestESMCProteinFunctionGraph(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()

    def test_load_gene_ontology_with_valid_dag(self):
        graph = DiGraph()
        graph.add_edge("GO:000001", "GO:000002")

        self.model.load_gene_ontology(graph)

        self.assertIs(self.model.graph, graph)

    def test_load_gene_ontology_with_cyclic_graph_raises_assertion_error(self):
        graph = DiGraph()
        graph.add_edge("GO:000001", "GO:000002")
        graph.add_edge("GO:000002", "GO:000001")

        with self.assertRaises(AssertionError):
            self.model.load_gene_ontology(graph)


class TestESMCProteinFunctionForward(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.batch_size = 2
        self.seq_len = 10
        self.x = torch.randint(0, 100, (self.batch_size, self.seq_len))

        self.mock_output = MagicMock()
        self.mock_output.embeddings = torch.randn(
            self.batch_size, self.seq_len, EMBEDDING_DIM
        )
        self.model.encoder.forward.return_value = self.mock_output

    def test_forward_mf_returns_correct_shape(self):
        out = self.model.forward_mf(self.x)
        self.assertEqual(out.shape, (self.batch_size, NUM_MF_CLASSES))

    def test_forward_bp_returns_correct_shape(self):
        out = self.model.forward_bp(self.x)
        self.assertEqual(out.shape, (self.batch_size, NUM_BP_CLASSES))

    def test_forward_cc_returns_correct_shape(self):
        out = self.model.forward_cc(self.x)
        self.assertEqual(out.shape, (self.batch_size, NUM_CC_CLASSES))

    def test_forward_all_returns_tuple_of_three_tensors(self):
        mf, bp, cc = self.model.forward_all(self.x)

        self.assertIsInstance(mf, torch.Tensor)
        self.assertIsInstance(bp, torch.Tensor)
        self.assertIsInstance(cc, torch.Tensor)
        self.assertEqual(mf.shape, (self.batch_size, NUM_MF_CLASSES))
        self.assertEqual(bp.shape, (self.batch_size, NUM_BP_CLASSES))
        self.assertEqual(cc.shape, (self.batch_size, NUM_CC_CLASSES))

    def test_forward_mf_output_dtype(self):
        out = self.model.forward_mf(self.x)
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_mf_calls_encoder_with_correct_args(self):
        self.model.forward_mf(self.x)
        self.model.encoder.forward.assert_called_once_with(
            sequence_tokens=self.x, sequence_id=None
        )

    def test_forward_bp_calls_encoder_with_correct_args(self):
        self.model.forward_bp(self.x)
        self.model.encoder.forward.assert_called_once_with(
            sequence_tokens=self.x, sequence_id=None
        )

    def test_forward_cc_calls_encoder_with_correct_args(self):
        self.model.forward_cc(self.x)
        self.model.encoder.forward.assert_called_once_with(
            sequence_tokens=self.x, sequence_id=None
        )

    def test_forward_with_single_sample(self):
        x = torch.randint(0, 100, (1, self.seq_len))
        mock_output = MagicMock()
        mock_output.embeddings = torch.randn(1, self.seq_len, EMBEDDING_DIM)
        self.model.encoder.forward.return_value = mock_output

        out = self.model.forward_mf(x)
        self.assertEqual(out.shape, (1, NUM_MF_CLASSES))


class TestESMCProteinFunctionPredict(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.batch_size = 2
        self.seq_len = 10
        self.x = torch.randint(0, 100, (self.batch_size, self.seq_len))

    def test_predict_mf_applies_sigmoid(self):
        logits = torch.randn(self.batch_size, NUM_MF_CLASSES) * 3

        with patch.object(self.model, "forward_mf", return_value=logits):
            probs = self.model.predict_mf(self.x)
            expected = torch.sigmoid(logits)

            self.assertTrue(torch.allclose(probs, expected))
            self.assertTrue(torch.all(probs >= 0).item())
            self.assertTrue(torch.all(probs <= 1).item())

    def test_predict_bp_applies_sigmoid(self):
        logits = torch.randn(self.batch_size, NUM_BP_CLASSES) * 3

        with patch.object(self.model, "forward_bp", return_value=logits):
            probs = self.model.predict_bp(self.x)
            expected = torch.sigmoid(logits)

            self.assertTrue(torch.allclose(probs, expected))

    def test_predict_cc_applies_sigmoid(self):
        logits = torch.randn(self.batch_size, NUM_CC_CLASSES) * 3

        with patch.object(self.model, "forward_cc", return_value=logits):
            probs = self.model.predict_cc(self.x)
            expected = torch.sigmoid(logits)

            self.assertTrue(torch.allclose(probs, expected))

    def test_predict_all_applies_sigmoid(self):
        mf_logits = torch.randn(self.batch_size, NUM_MF_CLASSES) * 3
        bp_logits = torch.randn(self.batch_size, NUM_BP_CLASSES) * 3
        cc_logits = torch.randn(self.batch_size, NUM_CC_CLASSES) * 3

        with patch.object(
            self.model, "forward_all", return_value=(mf_logits, bp_logits, cc_logits)
        ):
            mf_probs, bp_probs, cc_probs = self.model.predict_all(self.x)

            self.assertTrue(torch.allclose(mf_probs, torch.sigmoid(mf_logits)))
            self.assertTrue(torch.allclose(bp_probs, torch.sigmoid(bp_logits)))
            self.assertTrue(torch.allclose(cc_probs, torch.sigmoid(cc_logits)))

    def test_predict_mf_output_is_probability(self):
        logits = torch.randn(self.batch_size, NUM_MF_CLASSES)

        with patch.object(self.model, "forward_mf", return_value=logits):
            probs = self.model.predict_mf(self.x)
            self.assertTrue(torch.all(probs >= 0).item())
            self.assertTrue(torch.all(probs <= 1).item())

    def test_predict_with_single_sample(self):
        x = torch.randint(0, 100, (1, self.seq_len))
        logits = torch.randn(1, NUM_MF_CLASSES)

        with patch.object(self.model, "forward_mf", return_value=logits):
            probs = self.model.predict_mf(x)
            self.assertEqual(probs.shape, (1, NUM_MF_CLASSES))


class TestESMCProteinFunctionPredictTerms(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.batch_size = 2
        self.seq_len = 10
        self.x = torch.randint(0, 100, (self.batch_size, self.seq_len))

    def test_predict_mf_terms_returns_list_of_dicts(self):
        probs = torch.tensor([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])

        with patch.object(self.model, "predict_mf", return_value=probs):
            terms = self.model.predict_mf_terms(self.x, top_p=0.5)

            self.assertIsInstance(terms, list)
            self.assertEqual(len(terms), self.batch_size)
            for sample in terms:
                self.assertIsInstance(sample, dict)
                for key, value in sample.items():
                    self.assertIsInstance(key, str)
                    self.assertIsInstance(value, float)

    def test_predict_mf_terms_filters_by_top_p(self):
        probs = torch.tensor([[0.9, 0.1, 0.8]])

        with patch.object(self.model, "predict_mf", return_value=probs):
            terms = self.model.predict_mf_terms(self.x, top_p=0.5)

            self.assertEqual(len(terms), 1)
            self.assertIn("GO:000000", terms[0])
            self.assertIn("GO:000002", terms[0])
            self.assertNotIn("GO:000001", terms[0])

    def test_predict_bp_terms_returns_list_of_dicts(self):
        probs = torch.tensor([[0.9, 0.1, 0.8]])

        with patch.object(self.model, "predict_bp", return_value=probs):
            terms = self.model.predict_bp_terms(self.x, top_p=0.5)

            self.assertIsInstance(terms, list)
            self.assertEqual(len(terms), 1)
            for sample in terms:
                self.assertIsInstance(sample, dict)

    def test_predict_cc_terms_returns_list_of_dicts(self):
        probs = torch.tensor([[0.9, 0.1, 0.8]])

        with patch.object(self.model, "predict_cc", return_value=probs):
            terms = self.model.predict_cc_terms(self.x, top_p=0.5)

            self.assertIsInstance(terms, list)
            self.assertEqual(len(terms), 1)

    def test_predict_all_terms_returns_tuple_of_lists(self):
        probs = torch.tensor([[0.9, 0.1, 0.8]])

        with patch.object(
            self.model,
            "predict_all",
            return_value=(probs, probs, probs),
        ):
            mf_terms, bp_terms, cc_terms = self.model.predict_all_terms(
                self.x, top_p=0.5
            )

            self.assertIsInstance(mf_terms, list)
            self.assertIsInstance(bp_terms, list)
            self.assertIsInstance(cc_terms, list)
            self.assertEqual(len(mf_terms), 1)
            self.assertEqual(len(bp_terms), 1)
            self.assertEqual(len(cc_terms), 1)

    def test_predict_terms_with_top_p_at_boundary_one(self):
        probs = torch.tensor([[0.5, 0.5, 0.5]])

        with patch.object(self.model, "predict_mf", return_value=probs):
            terms = self.model.predict_mf_terms(self.x, top_p=1.0)

            self.assertEqual(len(terms), 1)
            self.assertEqual(len(terms[0]), 0)

    def test_predict_terms_with_all_probabilities_above_threshold(self):
        probs = torch.tensor([[0.6, 0.7, 0.8]])

        with patch.object(self.model, "predict_mf", return_value=probs):
            terms = self.model.predict_mf_terms(self.x, top_p=0.5)

            self.assertEqual(len(terms[0]), 3)

    def test_predict_terms_with_no_probabilities_above_threshold(self):
        probs = torch.tensor([[0.1, 0.2, 0.3]])

        with patch.object(self.model, "predict_mf", return_value=probs):
            terms = self.model.predict_mf_terms(self.x, top_p=0.5)

            self.assertEqual(len(terms[0]), 0)


class TestESMCProteinFunctionPredictSubgraphs(unittest.TestCase):
    def setUp(self):
        self.model = _make_model()
        self.batch_size = 2
        self.seq_len = 10
        self.x = torch.randint(0, 100, (self.batch_size, self.seq_len))

        self.graph = DiGraph()
        self.graph.add_edge("GO:000000", "GO:000001")
        self.graph.add_edge("GO:000000", "GO:000002")
        self.graph.add_edge("GO:000010", "GO:000011")
        self.model.load_gene_ontology(self.graph)

    def test_predict_mf_subgraphs_returns_list_of_digraphs(self):
        terms = [
            {"GO:000000": 0.9, "GO:000001": 0.8},
            {"GO:000002": 0.7},
        ]

        with patch.object(self.model, "predict_mf_terms", return_value=terms):
            subgraphs = self.model.predict_mf_subgraphs(self.x, top_p=0.5)

            self.assertIsInstance(subgraphs, list)
            self.assertEqual(len(subgraphs), self.batch_size)
            for subgraph in subgraphs:
                self.assertIsInstance(subgraph, DiGraph)

    def test_predict_mf_subgraphs_nodes_have_probability_attribute(self):
        terms = [
            {"GO:000000": 0.9},
            {"GO:000010": 0.8},
        ]

        with patch.object(self.model, "predict_mf_terms", return_value=terms):
            subgraphs = self.model.predict_mf_subgraphs(self.x, top_p=0.5)

            for subgraph in subgraphs:
                for node in subgraph.nodes():
                    self.assertIn("probability", subgraph.nodes[node])
                    self.assertIsInstance(subgraph.nodes[node]["probability"], float)

    def test_predict_bp_subgraphs_returns_list_of_digraphs(self):
        terms = [{"GO:000000": 0.9}, {"GO:000001": 0.8}]

        with patch.object(self.model, "predict_bp_terms", return_value=terms):
            subgraphs = self.model.predict_bp_subgraphs(self.x, top_p=0.5)

            self.assertIsInstance(subgraphs, list)
            self.assertEqual(len(subgraphs), self.batch_size)
            for subgraph in subgraphs:
                self.assertIsInstance(subgraph, DiGraph)

    def test_predict_cc_subgraphs_returns_list_of_digraphs(self):
        terms = [{"GO:000000": 0.9}, {"GO:000001": 0.8}]

        with patch.object(self.model, "predict_cc_terms", return_value=terms):
            subgraphs = self.model.predict_cc_subgraphs(self.x, top_p=0.5)

            self.assertIsInstance(subgraphs, list)
            self.assertEqual(len(subgraphs), self.batch_size)
            for subgraph in subgraphs:
                self.assertIsInstance(subgraph, DiGraph)

    def test_predict_all_subgraphs_returns_tuple_of_lists(self):
        terms = [{"GO:000000": 0.9}]

        with patch.object(
            self.model,
            "predict_all_terms",
            return_value=(terms, terms, terms),
        ):
            mf_sg, bp_sg, cc_sg = self.model.predict_all_subgraphs(self.x, top_p=0.5)

            self.assertIsInstance(mf_sg, list)
            self.assertIsInstance(bp_sg, list)
            self.assertIsInstance(cc_sg, list)
            for sg_list in (mf_sg, bp_sg, cc_sg):
                for subgraph in sg_list:
                    self.assertIsInstance(subgraph, DiGraph)
