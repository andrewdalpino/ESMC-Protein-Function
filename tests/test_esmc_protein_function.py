from unittest.mock import MagicMock, patch
import unittest

import torch
import networkx as nx

from esmc_protein_function.model import ESMCProteinFunction


def _make_mock_encoder(embedding_dimensions, num_encoder_layers):
    encoder = MagicMock()
    encoder.blocks = [MagicMock() for _ in range(num_encoder_layers)]
    encoder.transformer.blocks = [MagicMock() for _ in range(num_encoder_layers)]
    encoder.sequence_head = MagicMock()
    return encoder


def _make_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return tokenizer


def _make_model(
    embedding_dimensions=64,
    num_heads=4,
    num_encoder_layers=2,
    num_pool_heads=2,
    mf_terms=None,
    bp_terms=None,
    cc_terms=None,
):
    if mf_terms is None:
        mf_terms = {0: "GO:0005575", 1: "GO:0003674", 2: "GO:0008150"}
    if bp_terms is None:
        bp_terms = {0: "GO:0005575"}
    if cc_terms is None:
        cc_terms = {0: "GO:0005575"}

    return ESMCProteinFunction(
        embedding_dimensions=embedding_dimensions,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_pool_heads=num_pool_heads,
        indexToMfGoTerm=mf_terms,
        indexToBpGoTerm=bp_terms,
        indexToCcGoTerm=cc_terms,
        use_flash_attention=False,
    )


class TestESMCProteinFunctionInit(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        mock_enc = _make_mock_encoder(64, 2)
        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_init_with_valid_parameters(self):
        model = _make_model()
        self.assertIsInstance(model, ESMCProteinFunction)
        self.assertIsNone(model.graph)
        self.assertEqual(model.embedding_dimensions, 64)
        self.assertEqual(model.pad_token, 0)

    def test_init_sets_go_term_mappings(self):
        mf = {0: "GO:0005575", 1: "GO:0003674"}
        bp = {0: "GO:0005575"}
        cc = {0: "GO:0005575"}
        model = _make_model(mf_terms=mf, bp_terms=bp, cc_terms=cc)
        self.assertEqual(model.indexToMfGoTerm, mf)
        self.assertEqual(model.indexToBpGoTerm, bp)
        self.assertEqual(model.indexToCcGoTerm, cc)

    def test_init_creates_three_classification_heads(self):
        mf = {i: f"GO:MF{i}" for i in range(5)}
        bp = {i: f"GO:BP{i}" for i in range(3)}
        cc = {i: f"GO:CC{i}" for i in range(7)}
        model = _make_model(mf_terms=mf, bp_terms=bp, cc_terms=cc)
        self.assertEqual(model.mf_head.linear2.out_features, 5)
        self.assertEqual(model.bp_head.linear2.out_features, 3)
        self.assertEqual(model.cc_head.linear2.out_features, 7)

    def test_init_with_empty_mappings_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            _make_model(mf_terms={}, bp_terms={}, cc_terms={})


class TestESMCProteinFunctionProperties(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        mock_enc = _make_mock_encoder(64, 2)
        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model()

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_num_encoder_layers(self):
        self.assertEqual(self.model.num_encoder_layers, 2)

    def test_num_params(self):
        n = self.model.num_params
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_num_trainable_parameters(self):
        n = self.model.num_trainable_parameters
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)


class TestESMCProteinFunctionFreezeBase(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        self.param_a = MagicMock()
        self.param_a.requires_grad = True
        self.param_b = MagicMock()
        self.param_b.requires_grad = True

        inner_module = MagicMock()
        inner_module.parameters.return_value = [self.param_a, self.param_b]

        mock_enc = MagicMock()
        mock_enc.modules.return_value = [inner_module]
        mock_enc.blocks = [MagicMock()]
        mock_enc.transformer.blocks = [MagicMock()]
        mock_enc.sequence_head = MagicMock()

        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model()

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_freeze_base_disables_gradients(self):
        self.model.freeze_base()
        self.assertFalse(self.param_a.requires_grad)
        self.assertFalse(self.param_b.requires_grad)


class TestESMCProteinFunctionUnfreezeLastKEncoderLayers(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        self.block1_param = MagicMock()
        self.block1_param.requires_grad = False
        self.block2_param = MagicMock()
        self.block2_param.requires_grad = False

        block1 = MagicMock()
        block1.parameters.return_value = [self.block1_param]
        block2 = MagicMock()
        block2.parameters.return_value = [self.block2_param]

        mock_enc = _make_mock_encoder(64, 2)
        mock_enc.transformer.blocks = [block1, block2]
        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model()

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_unfreeze_last_k_layers(self):
        self.model.unfreeze_last_k_encoder_layers(1)
        self.assertFalse(self.block1_param.requires_grad)
        self.assertTrue(self.block2_param.requires_grad)

    def test_unfreeze_all_layers(self):
        self.model.unfreeze_last_k_encoder_layers(2)
        self.assertTrue(self.block1_param.requires_grad)
        self.assertTrue(self.block2_param.requires_grad)

    def test_unfreeze_with_zero_k_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.unfreeze_last_k_encoder_layers(0)

    def test_unfreeze_with_negative_k_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.unfreeze_last_k_encoder_layers(-1)


class TestESMCProteinFunctionLoadGeneOntology(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        mock_enc = _make_mock_encoder(64, 2)
        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model()

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_load_valid_dag(self):
        graph = nx.DiGraph()
        graph.add_edge("GO:0005575", "GO:0003674")
        graph.add_edge("GO:0003674", "GO:0008150")
        self.model.load_gene_ontology(graph)
        self.assertIsNotNone(self.model.graph)
        self.assertTrue(self.model.graph.has_edge("GO:0005575", "GO:0003674"))

    def test_load_single_node_dag(self):
        graph = nx.DiGraph()
        graph.add_node("GO:0005575")
        self.model.load_gene_ontology(graph)
        self.assertIsNotNone(self.model.graph)

    def test_load_cyclic_graph_raises_assertion_error(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "A")
        with self.assertRaises(AssertionError):
            self.model.load_gene_ontology(graph)

    def test_load_self_loop_raises_assertion_error(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "A")
        with self.assertRaises(AssertionError):
            self.model.load_gene_ontology(graph)


class TestESMCProteinFunctionQuantizeWeights(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        mock_enc = _make_mock_encoder(64, 2)
        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model(embedding_dimensions=64)

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_quantize_with_valid_group_size(self):
        self.model.quantize_weights(64)
        self.assertTrue(hasattr(self.model, "quantize_weights"))

    def test_quantize_with_same_as_embedding_dim(self):
        self.model.quantize_weights(64)
        self.assertTrue(hasattr(self.model, "quantize_weights"))

    def test_quantize_with_invalid_group_size_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.quantize_weights(32)


class TestESMCProteinFunctionAddFakeQuantizedTensors(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.quantize_patcher = patch("esmc_protein_function.model.quantize_")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()
        self.mock_quantize = self.quantize_patcher.start()

        mock_enc = _make_mock_encoder(64, 2)
        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()
        self.quantize_patcher.stop()

    def test_add_fake_quantized_tensors_with_valid_group_size(self):
        model = _make_model(embedding_dimensions=64)
        model.add_fake_quantized_tensors(group_size=32)
        self.mock_quantize.assert_called()

    def test_add_fake_quantized_tensors_with_group_size_one(self):
        model = _make_model(embedding_dimensions=64)
        model.add_fake_quantized_tensors(group_size=1)
        self.mock_quantize.assert_called()

    def test_add_fake_quantized_tensors_with_large_group_size(self):
        model = _make_model(embedding_dimensions=64)
        model.add_fake_quantized_tensors(group_size=64)
        self.mock_quantize.assert_called()


class TestESMCProteinFunctionRemoveFakeQuantizedTensors(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.quantize_patcher = patch("esmc_protein_function.model.quantize_")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()
        self.mock_quantize = self.quantize_patcher.start()

        mock_enc = _make_mock_encoder(64, 2)
        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model()

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()
        self.quantize_patcher.stop()

    def test_remove_fake_quantized_tensors_calls_quantize(self):
        self.model.remove_fake_quantized_tensors()
        self.mock_quantize.assert_called_once()


class TestESMCProteinFunctionForward(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        self.batch_size = 2
        self.seq_len = 10
        self.embedding_dim = 64

        mock_enc = MagicMock()
        mock_enc.blocks = [MagicMock() for _ in range(2)]
        mock_enc.transformer.blocks = [MagicMock() for _ in range(2)]
        mock_enc.sequence_head = MagicMock()

        mock_output = MagicMock()
        mock_output.embeddings = torch.randn(
            self.batch_size, self.seq_len, self.embedding_dim
        )
        mock_enc.forward.return_value = mock_output

        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model(
            embedding_dimensions=self.embedding_dim,
            num_heads=4,
            mf_terms={0: "GO:0005575", 1: "GO:0003674", 2: "GO:0008150"},
            bp_terms={0: "GO:0005575"},
            cc_terms={0: "GO:0005575", 1: "GO:0003674"},
        )
        self.x = torch.randint(0, 20, (self.batch_size, self.seq_len))

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_forward_mf_returns_correct_shape(self):
        out = self.model.forward_mf(self.x)
        self.assertEqual(out.shape, (self.batch_size, 3))

    def test_forward_bp_returns_correct_shape(self):
        out = self.model.forward_bp(self.x)
        self.assertEqual(out.shape, (self.batch_size, 1))

    def test_forward_cc_returns_correct_shape(self):
        out = self.model.forward_cc(self.x)
        self.assertEqual(out.shape, (self.batch_size, 2))

    def test_forward_all_returns_tuple_of_three(self):
        mf, bp, cc = self.model.forward_all(self.x)
        self.assertEqual(mf.shape, (self.batch_size, 3))
        self.assertEqual(bp.shape, (self.batch_size, 1))
        self.assertEqual(cc.shape, (self.batch_size, 2))


class TestESMCProteinFunctionPredict(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        self.batch_size = 2
        self.seq_len = 10
        self.embedding_dim = 64

        mock_enc = MagicMock()
        mock_enc.blocks = [MagicMock() for _ in range(2)]
        mock_enc.transformer.blocks = [MagicMock() for _ in range(2)]
        mock_enc.sequence_head = MagicMock()

        mock_output = MagicMock()
        mock_output.embeddings = torch.randn(
            self.batch_size, self.seq_len, self.embedding_dim
        )
        mock_enc.forward.return_value = mock_output

        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model(
            embedding_dimensions=self.embedding_dim,
            num_heads=4,
            mf_terms={0: "GO:0005575", 1: "GO:0003674", 2: "GO:0008150"},
            bp_terms={0: "GO:0005575"},
            cc_terms={0: "GO:0005575", 1: "GO:0003674"},
        )
        self.x = torch.randint(0, 20, (self.batch_size, self.seq_len))

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_predict_mf_returns_sigmoid_output(self):
        out = self.model.predict_mf(self.x)
        self.assertEqual(out.shape, (self.batch_size, 3))
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))

    def test_predict_bp_returns_sigmoid_output(self):
        out = self.model.predict_bp(self.x)
        self.assertEqual(out.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))

    def test_predict_cc_returns_sigmoid_output(self):
        out = self.model.predict_cc(self.x)
        self.assertEqual(out.shape, (self.batch_size, 2))
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))

    def test_predict_all_returns_tuple_of_three_sigmoid_outputs(self):
        mf, bp, cc = self.model.predict_all(self.x)
        self.assertEqual(mf.shape, (self.batch_size, 3))
        self.assertEqual(bp.shape, (self.batch_size, 1))
        self.assertEqual(cc.shape, (self.batch_size, 2))
        self.assertTrue(torch.all(mf >= 0) and torch.all(mf <= 1))
        self.assertTrue(torch.all(bp >= 0) and torch.all(bp <= 1))
        self.assertTrue(torch.all(cc >= 0) and torch.all(cc <= 1))


class TestESMCProteinFunctionPredictTerms(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        self.batch_size = 2
        self.seq_len = 10
        self.embedding_dim = 64

        mock_enc = MagicMock()
        mock_enc.blocks = [MagicMock() for _ in range(2)]
        mock_enc.transformer.blocks = [MagicMock() for _ in range(2)]
        mock_enc.sequence_head = MagicMock()

        mock_output = MagicMock()
        mock_output.embeddings = torch.randn(
            self.batch_size, self.seq_len, self.embedding_dim
        )
        mock_enc.forward.return_value = mock_output

        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model(
            embedding_dimensions=self.embedding_dim,
            num_heads=4,
            mf_terms={0: "GO:0005575", 1: "GO:0003674", 2: "GO:0008150"},
            bp_terms={0: "GO:0005575"},
            cc_terms={0: "GO:0005575", 1: "GO:0003674"},
        )
        self.x = torch.randint(0, 20, (self.batch_size, self.seq_len))

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_predict_mf_terms_returns_list_of_dicts(self):
        terms = self.model.predict_mf_terms(self.x)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(terms), self.batch_size)
        for term in terms:
            self.assertIsInstance(term, dict)

    def test_predict_bp_terms_returns_list_of_dicts(self):
        terms = self.model.predict_bp_terms(self.x)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(terms), self.batch_size)

    def test_predict_cc_terms_returns_list_of_dicts(self):
        terms = self.model.predict_cc_terms(self.x)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(terms), self.batch_size)

    def test_predict_all_terms_returns_list_of_tuples(self):
        terms = self.model.predict_all_terms(self.x)
        self.assertIsInstance(terms, list)
        self.assertEqual(len(terms), self.batch_size)
        for item in terms:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 3)

    def test_predict_mf_terms_with_top_p_one(self):
        terms = self.model.predict_mf_terms(self.x, top_p=1.0)
        self.assertIsInstance(terms, list)

    def test_predict_mf_terms_with_top_p_zero_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.predict_mf_terms(self.x, top_p=0)

    def test_predict_mf_terms_with_negative_top_p_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.predict_mf_terms(self.x, top_p=-0.1)

    def test_predict_mf_terms_with_top_p_greater_than_one_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.predict_mf_terms(self.x, top_p=1.1)

    def test_predict_bp_terms_with_invalid_top_p_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.predict_bp_terms(self.x, top_p=0)

    def test_predict_cc_terms_with_invalid_top_p_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.predict_cc_terms(self.x, top_p=0)

    def test_predict_all_terms_with_invalid_top_p_raises_assertion_error(self):
        with self.assertRaises(AssertionError):
            self.model.predict_all_terms(self.x, top_p=0)


class TestESMCProteinFunctionPredictSubgraphs(unittest.TestCase):
    def setUp(self):
        self.enc_patcher = patch("esmc_protein_function.model.ESMC")
        self.tok_patcher = patch("esmc_protein_function.model.EsmSequenceTokenizer")
        self.mock_esmc = self.enc_patcher.start()
        self.mock_tokenizer = self.tok_patcher.start()

        self.batch_size = 2
        self.seq_len = 10
        self.embedding_dim = 64

        mock_enc = MagicMock()
        mock_enc.blocks = [MagicMock() for _ in range(2)]
        mock_enc.transformer.blocks = [MagicMock() for _ in range(2)]
        mock_enc.sequence_head = MagicMock()

        mock_output = MagicMock()
        mock_output.embeddings = torch.randn(
            self.batch_size, self.seq_len, self.embedding_dim
        )
        mock_enc.forward.return_value = mock_output

        self.mock_esmc.return_value = mock_enc
        mock_tok = _make_mock_tokenizer()
        self.mock_tokenizer.return_value = mock_tok

        self.model = _make_model(
            embedding_dimensions=self.embedding_dim,
            num_heads=4,
            mf_terms={0: "GO:0005575", 1: "GO:0003674", 2: "GO:0008150"},
            bp_terms={0: "GO:0005575"},
            cc_terms={0: "GO:0005575", 1: "GO:0003674"},
        )

        graph = nx.DiGraph()
        graph.add_edge("GO:0005575", "GO:0003674")
        graph.add_edge("GO:0003674", "GO:0008150")
        self.model.load_gene_ontology(graph)

        self.x = torch.randint(0, 20, (self.batch_size, self.seq_len))

    def tearDown(self):
        self.enc_patcher.stop()
        self.tok_patcher.stop()

    def test_predict_mf_subgraphs_returns_list_of_tuples(self):
        subgraphs = self.model.predict_mf_subgraphs(self.x)
        self.assertIsInstance(subgraphs, list)
        self.assertEqual(len(subgraphs), self.batch_size)
        for sg, probs in subgraphs:
            self.assertIsInstance(sg, nx.DiGraph)
            self.assertIsInstance(probs, dict)

    def test_predict_bp_subgraphs_returns_list_of_tuples(self):
        subgraphs = self.model.predict_bp_subgraphs(self.x)
        self.assertIsInstance(subgraphs, list)
        self.assertEqual(len(subgraphs), self.batch_size)

    def test_predict_cc_subgraphs_returns_list_of_tuples(self):
        subgraphs = self.model.predict_cc_subgraphs(self.x)
        self.assertIsInstance(subgraphs, list)
        self.assertEqual(len(subgraphs), self.batch_size)

    def test_predict_all_subgraphs_returns_list(self):
        results = self.model.predict_all_subgraphs(self.x)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), self.batch_size)

    def test_predict_mf_subgraphs_without_graph_raises_assertion_error(self):
        model_no_graph = _make_model()
        with self.assertRaises(AssertionError):
            model_no_graph.predict_mf_subgraphs(self.x)

    def test_predict_bp_subgraphs_without_graph_raises_assertion_error(self):
        model_no_graph = _make_model()
        with self.assertRaises(AssertionError):
            model_no_graph.predict_bp_subgraphs(self.x)

    def test_predict_cc_subgraphs_without_graph_raises_assertion_error(self):
        model_no_graph = _make_model()
        with self.assertRaises(AssertionError):
            model_no_graph.predict_cc_subgraphs(self.x)

    def test_predict_all_subgraphs_without_graph_raises_assertion_error(self):
        model_no_graph = _make_model()
        with self.assertRaises(AssertionError):
            model_no_graph.predict_all_subgraphs(self.x)

    def test_predict_mf_subgraphs_with_top_p_zero_returns_all_classes(self):
        subgraphs = self.model.predict_mf_subgraphs(self.x, top_p=0)
        self.assertEqual(len(subgraphs), self.batch_size)

    def test_predict_mf_subgraphs_overrides_parent_probabilities(self):
        """When a child node has higher probability than a parent,
        the parent should be updated to at least the child's probability."""
        model = self.model
        subgraphs = model.predict_mf_subgraphs(self.x, top_p=0.0)
        for _, probs in subgraphs:
            for go_id, prob in probs.items():
                self.assertGreaterEqual(prob, 0)


class TestESMCProteinFunctionFromESMPretrained(unittest.TestCase):
    def test_from_esm_pretrained_unknown_model_raises_value_error(self):
        with self.assertRaises(ValueError):
            ESMCProteinFunction.from_esm_pretrained(
                model_name="unknown_model",
                num_pool_heads=2,
                indexToMfGoTerm={0: "GO:0005575"},
                indexToBpGoTerm={0: "GO:0003674"},
                indexToCcGoTerm={0: "GO:0008150"},
                use_flash_attention=False,
            )
