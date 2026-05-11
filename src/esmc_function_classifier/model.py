from functools import partial
from collections import defaultdict

import torch

from torch import Tensor
from torch.nn import Module, Identity, Linear, LayerNorm, Softmax, Flatten

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


class ESMCGeneOntology(Module, PyTorchModelHubMixin):
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
        num_pool_heads: int,
        indexToMfGoTerm: dict[int, str],
        indexToBpGoTerm: dict[int, str],
        indexToCcGoTerm: dict[int, str],
        use_flash_attention: bool,
    ) -> "ESMCGeneOntology":
        """
        Since the base model pretrained weights are stored in a proprietary pickle format,
        let's implement a custom factory method to load those weights.
        """

        from esm.utils.constants.esm3 import data_root

        if model_name not in cls.ESM_PRETRAINED_CONFIGS:
            raise ValueError(f"Unknown model name: {model_name}")

        model_args = cls.ESM_PRETRAINED_CONFIGS.get(model_name)

        model = cls(
            embedding_dimensions=model_args["embedding_dimensions"],
            num_heads=model_args["num_heads"],
            num_encoder_layers=model_args["num_encoder_layers"],
            num_pool_heads=num_pool_heads,
            indexToMfGoTerm=indexToMfGoTerm,
            indexToBpGoTerm=indexToBpGoTerm,
            indexToCcGoTerm=indexToCcGoTerm,
            use_flash_attention=use_flash_attention,
        )

        checkpoint_path = cls.ESM_PRETRAINED_CHECKPOINT_PATHS.get(model_name)

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
        num_pool_heads: int,
        indexToMfGoTerm: dict[int, str],
        indexToBpGoTerm: dict[int, str],
        indexToCcGoTerm: dict[int, str],
        use_flash_attention: bool,
    ) -> None:
        super().__init__()

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

        num_mf_classes = len(indexToMfGoTerm)
        num_bp_classes = len(indexToBpGoTerm)
        num_cc_classes = len(indexToCcGoTerm)

        self.encoder = encoder

        new_classifier = partial(
            MultiLabelClassifier,
            embedding_dimensions=embedding_dimensions,
            num_heads=num_pool_heads,
        )

        self.mf_head = new_classifier(num_classes=num_mf_classes)
        self.bp_head = new_classifier(num_classes=num_bp_classes)
        self.cc_head = new_classifier(num_classes=num_cc_classes)

        self.indexToMfGoTerm = indexToMfGoTerm
        self.indexToBpGoTerm = indexToBpGoTerm
        self.indexToCcGoTerm = indexToCcGoTerm

        self.embedding_dimensions = embedding_dimensions
        self.pad_token = tokenizer.pad_token_id

        self.graph: DiGraph | None = None

        self.tokenizer = tokenizer

    @property
    def num_encoder_layers(self) -> int:
        return len(self.encoder.blocks)

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

        assert group_size % self.embedding_dimensions == 0, "Invalid quant group size."

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

        z = self.mf_head.forward(out.embeddings)

        return z

    def forward_bp(self, x: Tensor) -> Tensor:
        out: ESMCOutput = self.encoder.forward(sequence_tokens=x, sequence_id=None)

        z = self.bp_head.forward(out.embeddings)

        return z

    def forward_cc(self, x: Tensor) -> Tensor:
        out: ESMCOutput = self.encoder.forward(sequence_tokens=x, sequence_id=None)

        z = self.cc_head.forward(out.embeddings)

        return z

    def forward_all(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out: ESMCOutput = self.encoder.forward(sequence_tokens=x, sequence_id=None)

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

        assert 0 < top_p <= 1, "top_p must be in the range (0, 1]."

        z_prob = self.predict_mf(x)

        results = [
            {
                self.indexToMfGoTerm[index]: prob.item()
                for index, prob in enumerate(sample_probs)
                if prob > top_p
            }
            for sample_probs in z_prob
        ]

        return results

    @torch.inference_mode()
    def predict_bp_terms(self, x: Tensor, top_p: float = 0.5) -> list[dict[str, float]]:
        """Predicts BP GO terms based on the input sequence tokens."""

        assert 0 < top_p <= 1, "top_p must be in the range (0, 1]."

        z_prob = self.predict_bp(x)

        results = [
            {
                self.indexToBpGoTerm[index]: prob.item()
                for index, prob in enumerate(sample_probs)
                if prob > top_p
            }
            for sample_probs in z_prob
        ]

        return results

    @torch.inference_mode()
    def predict_cc_terms(self, x: Tensor, top_p: float = 0.5) -> list[dict[str, float]]:
        """Predicts CC GO terms based on the input sequence tokens."""

        assert 0 < top_p <= 1, "top_p must be in the range (0, 1]."

        z_prob = self.predict_cc(x)

        results = [
            {
                self.indexToCcGoTerm[index]: prob.item()
                for index, prob in enumerate(sample_probs)
                if prob > top_p
            }
            for sample_probs in z_prob
        ]

        return results

    @torch.inference_mode()
    def predict_all_terms(
        self, x: Tensor, top_p: float = 0.5
    ) -> list[tuple[dict[str, float], ...]]:
        """Predicts GO terms based on the input sequence tokens."""

        assert 0 < top_p <= 1, "top_p must be in the range (0, 1]."

        mf_prob, bp_prob, cc_prob = self.predict_all(x)

        aspects = [
            (self.indexToMfGoTerm, mf_prob),
            (self.indexToBpGoTerm, bp_prob),
            (self.indexToCcGoTerm, cc_prob),
        ]

        batch_size = mf_prob.shape[0]

        results = [
            tuple(
                {
                    mapping[index]: prob.item()
                    for index, prob in enumerate(probs[i])
                    if prob > top_p
                }
                for mapping, probs in aspects
            )
            for i in range(batch_size)
        ]

        return results

    @torch.inference_mode()
    def predict_mf_subgraphs(
        self, x: Tensor, top_p: float = 0.5
    ) -> list[tuple[DiGraph, dict[str, float]]]:
        """Predicts a subgraph of the MF aspect of the GO based on the input sequence tokens."""

        assert self.graph is not None, "Gene Ontology graph is not loaded."

        mf_prob = self.predict_mf(x)

        batch_size = mf_prob.shape[0]

        results = []

        for i in range(batch_size):
            sample_probs = mf_prob[i]

            child_nodes = {
                self.indexToMfGoTerm[index]: prob.item()
                for index, prob in enumerate(sample_probs)
                if prob > top_p
            }

            probabilities = defaultdict(float, child_nodes)

            # Fix up the predictions by leveraging the GO DAG hierarchy.
            for go_id, child_probability in child_nodes.items():
                for descendant in descendants(self.graph, go_id):
                    parent_probability = probabilities[descendant]

                    probabilities[descendant] = max(
                        parent_probability,
                        child_probability,
                    )

            subgraph = self.graph.subgraph(probabilities.keys())

            results.append((subgraph, dict(probabilities)))

        return results

    @torch.inference_mode()
    def predict_bp_subgraphs(
        self, x: Tensor, top_p: float = 0.5
    ) -> list[tuple[DiGraph, dict[str, float]]]:
        """Predicts a subgraph of the BP aspect of the GO based on the input sequence tokens."""

        assert self.graph is not None, "Gene Ontology graph is not loaded."

        bp_prob = self.predict_bp(x)

        batch_size = bp_prob.shape[0]

        results = []

        for i in range(batch_size):
            sample_probs = bp_prob[i]

            child_nodes = {
                self.indexToBpGoTerm[index]: prob.item()
                for index, prob in enumerate(sample_probs)
                if prob > top_p
            }

            probabilities = defaultdict(float, child_nodes)

            # Fix up the predictions by leveraging the GO DAG hierarchy.
            for go_id, child_probability in child_nodes.items():
                for descendant in descendants(self.graph, go_id):
                    parent_probability = probabilities[descendant]

                    probabilities[descendant] = max(
                        parent_probability,
                        child_probability,
                    )

            subgraph = self.graph.subgraph(probabilities.keys())

            results.append((subgraph, dict(probabilities)))

        return results

    @torch.inference_mode()
    def predict_cc_subgraphs(
        self, x: Tensor, top_p: float = 0.5
    ) -> list[tuple[DiGraph, dict[str, float]]]:
        """Predicts a subgraph of the CC aspect of the GO based on the input sequence tokens."""

        assert self.graph is not None, "Gene Ontology graph is not loaded."

        cc_prob = self.predict_cc(x)

        batch_size = cc_prob.shape[0]

        results = []

        for i in range(batch_size):
            sample_probs = cc_prob[i]

            child_nodes = {
                self.indexToCcGoTerm[index]: prob.item()
                for index, prob in enumerate(sample_probs)
                if prob > top_p
            }

            probabilities = defaultdict(float, child_nodes)

            # Fix up the predictions by leveraging the GO DAG hierarchy.
            for go_id, child_probability in child_nodes.items():
                for descendant in descendants(self.graph, go_id):
                    parent_probability = probabilities[descendant]

                    probabilities[descendant] = max(
                        parent_probability,
                        child_probability,
                    )

            subgraph = self.graph.subgraph(probabilities.keys())

            results.append((subgraph, dict(probabilities)))

        return results

    @torch.inference_mode()
    def predict_all_subgraphs(
        self, x: Tensor, top_p: float = 0.5
    ) -> list[tuple[list[DiGraph], list[dict[str, float]]]]:
        """Predicts a subgraph of the GO based on the input sequence tokens."""

        assert self.graph is not None, "Gene Ontology graph is not loaded."

        mf_prob, bp_prob, cc_prob = self.predict_all(x)

        batch_size = mf_prob.shape[0]

        aspects = [
            (self.indexToMfGoTerm, mf_prob),
            (self.indexToBpGoTerm, bp_prob),
            (self.indexToCcGoTerm, cc_prob),
        ]

        results = []

        for i in range(batch_size):
            subgraphs = []
            terms = []

            for mapping, probs in aspects:
                sample_probs = probs[i]

                child_nodes = {
                    mapping[index]: prob.item()
                    for index, prob in enumerate(sample_probs)
                    if prob > top_p
                }

                probabilities = defaultdict(float, child_nodes)

                # Fix up the predictions by leveraging the GO DAG hierarchy.
                for go_id, child_probability in child_nodes.items():
                    for descendant in descendants(self.graph, go_id):
                        parent_probability = probabilities[descendant]

                        probabilities[descendant] = max(
                            parent_probability,
                            child_probability,
                        )

                subgraph = self.graph.subgraph(probabilities.keys())

                subgraphs.append(subgraph)
                terms.append(dict(probabilities))

            results.append((subgraphs, terms))

        return results


class MultiLabelClassifier(Module):
    """A 2-layer multi-label binary classification head with SwiGLU activation."""

    def __init__(self, embedding_dimensions: int, num_heads: int, num_classes: int):
        super().__init__()

        assert embedding_dimensions > 0, "embedding_dimensions must be greater than 0."
        assert num_classes > 0, "num_classes must be greater than 0."

        self.pool = AttentionPool(embedding_dimensions, num_heads)

        self.linear1 = Linear(embedding_dimensions, 2 * embedding_dimensions)
        self.linear2 = Linear(embedding_dimensions, num_classes, bias=False)

        self.norm = LayerNorm(embedding_dimensions)

        self.swiglu = SwiGLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool.forward(x)

        z = self.linear1.forward(x)
        z = self.swiglu.forward(z)

        z = x + z

        z = self.norm.forward(z)
        z = self.linear2.forward(z)

        return z


class AttentionPool(Module):
    """
    A pooling layer that combines token embeddings into a single vector representation
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
