from collections import defaultdict

import torch

from torch import Tensor
from torch.nn import Module, Identity, Linear, LayerNorm, Sequential, Softmax, Flatten

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from torchao.quantization.qat import (
    FakeQuantizeConfig,
    IntXQuantizationAwareTrainingConfig,
    FromIntXQuantizationAwareTrainingConfig,
)

from esm.tokenization import EsmSequenceTokenizer
from esm.models.esmc import ESMC, ESMCOutput
from esm.layers.blocks import SwiGLU

from huggingface_hub import PyTorchModelHubMixin

from networkx import DiGraph, is_directed_acyclic_graph, descendants


class ESMCProteinFunction(Module, PyTorchModelHubMixin):
    """
    A model for predicting the Gene Ontology (GO) subgraph from protein sequences using the
    ESMC base model as an encoder.
    """

    ESM_PRETRAINED_CONFIGS = {
        "esmc_300m": {
            "embedding_dimensions": 960,
            "num_heads": 15,
            "num_encoder_layers": 30,
        },
        "esmc_600m": {
            "embedding_dimensions": 1152,
            "num_heads": 18,
            "num_encoder_layers": 36,
        },
    }

    ESM_PRETRAINED_CHECKPOINT_PATHS = {
        "esmc_300m": "data/weights/esmc_300m_2024_12_v0.pth",
        "esmc_600m": "data/weights/esmc_600m_2024_12_v0.pth",
    }

    @classmethod
    def from_esm_pretrained(
        cls,
        model_name: str,
        num_mf_pool_heads: int,
        num_bp_pool_heads: int,
        num_cc_pool_heads: int,
        num_mf_layers: int,
        num_bp_layers: int,
        num_cc_layers: int,
        index_to_mf_term: dict[int, str],
        index_to_bp_term: dict[int, str],
        index_to_cc_term: dict[int, str],
        use_flash_attention: bool,
    ) -> "ESMCProteinFunction":
        """
        Since the base model pretrained weights are stored in a proprietary pickle format,
        let's implement a custom factory method to load those weights.
        """

        from esm.utils.constants.esm3 import data_root

        model_args = cls.ESM_PRETRAINED_CONFIGS.get(model_name)
        checkpoint_path = cls.ESM_PRETRAINED_CHECKPOINT_PATHS.get(model_name)

        assert (
            model_args is not None
        ), f"Model args not found for model name: {model_name}."

        assert (
            checkpoint_path is not None
        ), f"Checkpoint path not found for model name: {model_name}."

        model = cls(
            embedding_dimensions=model_args["embedding_dimensions"],
            num_heads=model_args["num_heads"],
            num_encoder_layers=model_args["num_encoder_layers"],
            num_mf_pool_heads=num_mf_pool_heads,
            num_bp_pool_heads=num_bp_pool_heads,
            num_cc_pool_heads=num_cc_pool_heads,
            num_mf_layers=num_mf_layers,
            num_bp_layers=num_bp_layers,
            num_cc_layers=num_cc_layers,
            index_to_mf_term=index_to_mf_term,
            index_to_bp_term=index_to_bp_term,
            index_to_cc_term=index_to_cc_term,
            use_flash_attention=use_flash_attention,
        )

        # Compensate for irregular base model naming conventions.
        esm_model_name = model_name.replace("_", "-")

        checkpoint_path = data_root(esm_model_name) / checkpoint_path

        state_dict = torch.load(checkpoint_path)

        model.encoder.load_state_dict(state_dict, strict=False)

        return model

    def __init__(
        self,
        embedding_dimensions: int,
        num_heads: int,
        num_encoder_layers: int,
        num_mf_pool_heads: int,
        num_bp_pool_heads: int,
        num_cc_pool_heads: int,
        num_mf_layers: int,
        num_bp_layers: int,
        num_cc_layers: int,
        index_to_mf_term: dict[int, str],
        index_to_bp_term: dict[int, str],
        index_to_cc_term: dict[int, str],
        use_flash_attention: bool,
    ) -> None:
        super().__init__()

        assert index_to_bp_term, "index_to_bp_term must be non-empty."
        assert index_to_mf_term, "index_to_mf_term must be non-empty."
        assert index_to_cc_term, "index_to_cc_term must be non-empty."

        tokenizer = EsmSequenceTokenizer()

        encoder = ESMC(
            d_model=embedding_dimensions,
            n_heads=num_heads,
            n_layers=num_encoder_layers,
            tokenizer=tokenizer,
            use_flash_attn=use_flash_attention,
        )

        # Remove pretrained sequence head from the encoder.
        encoder.sequence_head = Identity()

        num_mf_classes = len(index_to_mf_term)
        num_bp_classes = len(index_to_bp_term)
        num_cc_classes = len(index_to_cc_term)

        self.encoder = encoder

        self.mf_head = GOTermClassifier(
            embedding_dimensions, num_mf_pool_heads, num_mf_layers, num_mf_classes
        )

        self.bp_head = GOTermClassifier(
            embedding_dimensions, num_bp_pool_heads, num_bp_layers, num_bp_classes
        )

        self.cc_head = GOTermClassifier(
            embedding_dimensions, num_cc_pool_heads, num_cc_layers, num_cc_classes
        )

        self.graph: DiGraph | None = None

        self.index_to_mf_term = index_to_mf_term
        self.index_to_bp_term = index_to_bp_term
        self.index_to_cc_term = index_to_cc_term
        self.embedding_dimensions = embedding_dimensions
        self.tokenizer = tokenizer

    @property
    def num_encoder_layers(self) -> int:
        return len(self.encoder.transformer.blocks)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_base(self) -> None:
        """Prevent the base model parameters from being updated during training."""

        for module in self.encoder.modules():
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_last_k_encoder_layers(self, k: int) -> None:
        """Allow the last k encoder layers to be trainable."""

        assert k > 0, "k must be greater than 0."

        for module in self.encoder.transformer.blocks[-k:]:
            for param in module.parameters():
                param.requires_grad = True

    def add_fake_quantized_tensors(self, group_size: int) -> None:
        """Prepare the model for quantization-aware training."""

        for module in self.modules():
            if isinstance(module, Linear):
                assert module.in_features % group_size == 0, (
                    f"quant_group_size ({group_size}) must divide in_features ({module.in_features})"
                    f" of layer {module}."
                )

        weight_config = FakeQuantizeConfig(torch.int8, group_size=group_size)

        config = IntXQuantizationAwareTrainingConfig(weight_config=weight_config)

        quantize_(self, config)

    def remove_fake_quantized_tensors(self) -> None:
        """Convert fake quantized tensors back to regular tensors."""

        config = FromIntXQuantizationAwareTrainingConfig()

        quantize_(self, config)

    def quantize_weights(self, group_size: int) -> None:
        """Quantize the weights of the model."""

        assert group_size % self.embedding_dimensions == 0, "Invalid quant group size."

        config = Int8WeightOnlyConfig(group_size=group_size)

        quantize_(self, config)

    def load_gene_ontology(self, graph: DiGraph) -> None:
        """Load the Gene Ontology (GO) DAG."""

        assert is_directed_acyclic_graph(
            graph
        ), "Invalid GO graph, must be a directed acyclic graph (DAG)."

        self.graph = graph

    def forward_mf(self, x: Tensor) -> Tensor:
        out: ESMCOutput = self.encoder.forward(sequence_tokens=x, sequence_id=None)

        assert out.embeddings is not None, "Missing encoder contextual embeddings."

        z = self.mf_head.forward(out.embeddings)

        return z

    def forward_bp(self, x: Tensor) -> Tensor:
        out: ESMCOutput = self.encoder.forward(sequence_tokens=x, sequence_id=None)

        assert out.embeddings is not None, "Missing encoder contextual embeddings."

        z = self.bp_head.forward(out.embeddings)

        return z

    def forward_cc(self, x: Tensor) -> Tensor:
        out: ESMCOutput = self.encoder.forward(sequence_tokens=x, sequence_id=None)

        assert out.embeddings is not None, "Missing encoder contextual embeddings."

        z = self.cc_head.forward(out.embeddings)

        return z

    def forward_all(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out: ESMCOutput = self.encoder.forward(sequence_tokens=x, sequence_id=None)

        assert out.embeddings is not None, "Missing encoder contextual embeddings."

        z_mf = self.mf_head.forward(out.embeddings)
        z_bp = self.bp_head.forward(out.embeddings)
        z_cc = self.cc_head.forward(out.embeddings)

        return z_mf, z_bp, z_cc

    @torch.inference_mode()
    def predict_mf(self, x: Tensor) -> Tensor:
        """Predicts MF GO terms based on the input sequence tokens."""

        z = self.forward_mf(x)

        z = torch.sigmoid(z)

        return z

    @torch.inference_mode()
    def predict_bp(self, x: Tensor) -> Tensor:
        """Predicts BP GO terms based on the input sequence tokens."""

        z = self.forward_bp(x)

        z = torch.sigmoid(z)

        return z

    @torch.inference_mode()
    def predict_cc(self, x: Tensor) -> Tensor:
        """Predicts CC GO terms based on the input sequence tokens."""

        z = self.forward_cc(x)

        z = torch.sigmoid(z)

        return z

    @torch.inference_mode()
    def predict_all(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predicts MF, BP, and CC GO terms based on the input sequence tokens."""

        z_mf, z_bp, z_cc = self.forward_all(x)

        z_mf = torch.sigmoid(z_mf)
        z_bp = torch.sigmoid(z_bp)
        z_cc = torch.sigmoid(z_cc)

        return z_mf, z_bp, z_cc

    @torch.inference_mode()
    def predict_mf_terms(self, x: Tensor, top_p: float = 0.5) -> list[dict[str, float]]:
        """Predicts MF GO terms based on the input sequence tokens."""

        probabilities = self.predict_mf(x)

        terms = self._match_terms(probabilities, self.index_to_mf_term, top_p)

        return terms

    @torch.inference_mode()
    def predict_bp_terms(self, x: Tensor, top_p: float = 0.5) -> list[dict[str, float]]:
        """Predicts BP GO terms based on the input sequence tokens."""

        probabilities = self.predict_bp(x)

        terms = self._match_terms(probabilities, self.index_to_bp_term, top_p)

        return terms

    @torch.inference_mode()
    def predict_cc_terms(self, x: Tensor, top_p: float = 0.5) -> list[dict[str, float]]:
        """Predicts CC GO terms based on the input sequence tokens."""

        probabilities = self.predict_cc(x)

        terms = self._match_terms(probabilities, self.index_to_cc_term, top_p)

        return terms

    @torch.inference_mode()
    def predict_all_terms(
        self, x: Tensor, top_p: float = 0.5
    ) -> tuple[list[dict[str, float]], ...]:
        """Predicts GO terms based on the input sequence tokens."""

        mf_prob, bp_prob, cc_prob = self.predict_all(x)

        mf_terms = self._match_terms(mf_prob, self.index_to_mf_term, top_p)
        bp_terms = self._match_terms(bp_prob, self.index_to_bp_term, top_p)
        cc_terms = self._match_terms(cc_prob, self.index_to_cc_term, top_p)

        return mf_terms, bp_terms, cc_terms

    @torch.inference_mode()
    def predict_mf_subgraphs(
        self, x: Tensor, top_p: float = 0.5
    ) -> tuple[list[DiGraph], list[dict[str, float]]]:
        """Predicts a subgraph of the MF aspect of the GO based on the input sequence tokens."""

        terms = self.predict_mf_terms(x, top_p)

        subgraphs, terms = self._build_subgraphs(terms)

        return subgraphs, terms

    @torch.inference_mode()
    def predict_bp_subgraphs(
        self, x: Tensor, top_p: float = 0.5
    ) -> tuple[list[DiGraph], list[dict[str, float]]]:
        """Predicts a subgraph of the BP aspect of the GO based on the input sequence tokens."""

        terms = self.predict_bp_terms(x, top_p)

        subgraphs, terms = self._build_subgraphs(terms)

        return subgraphs, terms

    @torch.inference_mode()
    def predict_cc_subgraphs(
        self, x: Tensor, top_p: float = 0.5
    ) -> tuple[list[DiGraph], list[dict[str, float]]]:
        """Predicts a subgraph of the CC aspect of the GO based on the input sequence tokens."""

        terms = self.predict_cc_terms(x, top_p)

        subgraphs, terms = self._build_subgraphs(terms)

        return subgraphs, terms

    @torch.inference_mode()
    def predict_all_subgraphs(
        self, x: Tensor, top_p: float = 0.5
    ) -> tuple[tuple[list[DiGraph], list[dict[str, float]]], ...]:
        """Predicts a subgraph of the GO based on the input sequence tokens."""

        aspects = self.predict_all_terms(x, top_p)

        results = []

        for terms in aspects:
            subgraphs, terms = self._build_subgraphs(terms)

            results.append((subgraphs, terms))

        return tuple(results)

    def _match_terms(
        self, probs: Tensor, mapping: dict[int, str], top_p: float
    ) -> list[dict[str, float]]:
        """
        Adds GO terms to the output based on the predicted probabilities and a specified threshold.
        """

        assert 0 < top_p <= 1, "top_p must be in the range (0, 1]."

        terms = []

        for sample_probs in probs:
            terms.append(
                {
                    mapping[index]: prob.item()
                    for index, prob in enumerate(sample_probs)
                    if prob > top_p
                }
            )

        return terms

    def _build_subgraphs(
        self, terms: list[dict[str, float]]
    ) -> tuple[list[DiGraph], list[dict[str, float]]]:
        """
        Builds subgraphs of the GO DAG based on the predicted probabilities.
        """

        assert self.graph is not None, "Gene Ontology graph is not loaded."

        subgraphs, probabilities = [], []

        for sample_terms in terms:
            term_probabilities = defaultdict(float, sample_terms)

            # Fix up the predictions by leveraging the GO DAG hierarchy.
            for go_id, child_probability in sample_terms.items():
                for descendant in descendants(self.graph, go_id):
                    parent_probability = term_probabilities[descendant]

                    term_probabilities[descendant] = max(
                        parent_probability,
                        child_probability,
                    )

            term_probabilities = dict(term_probabilities)

            subgraph = self.graph.subgraph(term_probabilities.keys())

            subgraphs.append(subgraph)
            probabilities.append(term_probabilities)

        return subgraphs, probabilities


class GOTermClassifier(Module):
    """
    A multi-label binary classification head for predicting GO terms from amino acid
    sequence contextual embeddings.
    """

    def __init__(
        self,
        embedding_dimensions: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()

        self.pool = AttentionPool(embedding_dimensions, num_heads)

        self.mlp = MLPClassifier(embedding_dimensions, num_layers, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        z = self.pool.forward(x)
        z = self.mlp.forward(z)

        return z


class AttentionPool(Module):
    """
    A pooling layer that combines contextual embeddings into a single vector representation
    using multi-headed attention.
    """

    def __init__(self, embedding_dimensions: int, num_heads: int):
        super().__init__()

        assert embedding_dimensions > 0, "embedding_dimensions must be greater than 0."
        assert num_heads > 0, "num_heads must be greater than 0."

        hidden_dimensions = num_heads * embedding_dimensions

        self.linear1 = Linear(embedding_dimensions, num_heads)
        self.linear2 = Linear(hidden_dimensions, embedding_dimensions)

        self.softmax = Softmax(dim=1)

        self.flatten = Flatten()

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1.forward(x)
        w = self.softmax.forward(z)

        z = (w.unsqueeze(2) * x.unsqueeze(-1)).sum(dim=1)

        z = self.flatten.forward(z)
        z = self.linear2.forward(z)

        return z


class MLPClassifier(Module):
    """
    A multi-layer perceptron (MLP) classification head.
    """

    def __init__(
        self,
        embedding_dimensions: int,
        num_layers: int,
        num_classes: int,
    ):
        super().__init__()

        assert embedding_dimensions > 0, "embedding_dimensions must be greater than 0."
        assert num_layers > 0, "num_layers must be greater than 0."
        assert num_classes > 0, "num_classes must be greater than 0."

        self.layers = Sequential(
            *[FeedForwardBlock(embedding_dimensions) for _ in range(num_layers)]
        )

        self.out = Linear(embedding_dimensions, num_classes, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.layers.forward(x)

        z = self.out.forward(z)

        return z


class FeedForwardBlock(Module):
    """
    A 2-layer feedforward block with a residual connection and SwiGLU activation.
    """

    def __init__(self, embedding_dimensions: int):
        super().__init__()

        assert embedding_dimensions > 0, "embedding_dimensions must be greater than 0."

        hidden_dimensions = 2 * embedding_dimensions

        self.linear = Linear(embedding_dimensions, hidden_dimensions)

        self.swiglu = SwiGLU()

        self.norm = LayerNorm(embedding_dimensions)

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear.forward(x)
        z = self.swiglu.forward(z)

        z = x + z

        z = self.norm.forward(z)

        return z
