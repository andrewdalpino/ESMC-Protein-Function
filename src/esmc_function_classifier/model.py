from copy import copy

from collections import defaultdict

import torch

from torch import Tensor
from torch.nn import Module, Identity, Linear

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


class EsmcGoTermClassifier(Module, PyTorchModelHubMixin):
    """
    A model for predicting Gene Ontology (GO) terms from protein sequences using the
    ESMC base model.
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
        indexToMfGoTerm: dict[int, str],
        indexToBpGoTerm: dict[int, str],
        indexToCcGoTerm: dict[int, str],
        use_flash_attention: bool,
    ) -> "EsmcGoTermClassifier":
        """
        Since the base model pretrained weights are stored in a proprietary pickle format,
        let's implement a custom factory method to load those weights.
        """

        from esm.utils.constants.esm3 import data_root

        if model_name not in cls.ESM_PRETRAINED_CONFIGS:
            raise ValueError(f"Unknown model name: {model_name}")

        model_args = cls.ESM_PRETRAINED_CONFIGS.get(model_name)

        # This is required for the base class but is not used otherwise.
        tokenizer = EsmSequenceTokenizer()

        encoder = ESMC(
            d_model=model_args["embedding_dimensions"],
            n_heads=model_args["num_heads"],
            n_layers=model_args["num_encoder_layers"],
            tokenizer=tokenizer,
            use_flash_attn=use_flash_attention,
        )

        checkpoint_path = cls.ESM_PRETRAINED_CHECKPOINT_PATHS.get(model_name)

        # Compensate for irregular base model naming conventions.
        esm_model_name = model_name.replace("_", "-")

        checkpoint_path = data_root(esm_model_name) / checkpoint_path

        state_dict = torch.load(checkpoint_path)

        encoder.load_state_dict(state_dict, strict=False)

        # Remove pretrained sequence head from the base model.
        encoder.sequence_head = Identity()

        model = cls(
            encoder,
            indexToMfGoTerm=indexToMfGoTerm,
            indexToBpGoTerm=indexToBpGoTerm,
            indexToCcGoTerm=indexToCcGoTerm,
        )

        return model

    def __init__(
        self,
        encoder: ESMC,
        indexToMfGoTerm: dict[int, str],
        indexToBpGoTerm: dict[int, str],
        indexToCcGoTerm: dict[int, str],
    ) -> None:
        super().__init__()

        num_mf_classes = len(indexToMfGoTerm)
        num_bp_classes = len(indexToBpGoTerm)
        num_cc_classes = len(indexToCcGoTerm)

        self.encoder = encoder

        self.mf_head = MultiLabelClassifier(encoder.embed.embedding_dim, num_mf_classes)
        self.bp_head = MultiLabelClassifier(encoder.embed.embedding_dim, num_bp_classes)
        self.cc_head = MultiLabelClassifier(encoder.embed.embedding_dim, num_cc_classes)

        self.indexToMfGoTerm = indexToMfGoTerm
        self.indexToBpGoTerm = indexToBpGoTerm
        self.indexToCcGoTerm = indexToCcGoTerm

        self.embedding_dimensions = encoder.embed.embedding_dim

        self.graph: DiGraph | None = None

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

    def forward_mf(
        self, sequence_tokens: Tensor, sequence_id: Tensor | None = None
    ) -> Tensor:
        out: ESMCOutput = self.encoder.forward(
            sequence_tokens=sequence_tokens,
            sequence_id=sequence_id,
        )

        # Grab the classification token <CLS> embeddings.
        x = out.embeddings[:, 0, :]

        z = self.mf_head.forward(x)

        return z

    def forward_bp(
        self, sequence_tokens: Tensor, sequence_id: Tensor | None = None
    ) -> Tensor:
        out: ESMCOutput = self.encoder.forward(
            sequence_tokens=sequence_tokens,
            sequence_id=sequence_id,
        )

        # Grab the classification token <CLS> embeddings.
        x = out.embeddings[:, 0, :]

        z = self.bp_head.forward(x)

        return z

    def forward_cc(
        self, sequence_tokens: Tensor, sequence_id: Tensor | None = None
    ) -> Tensor:
        out: ESMCOutput = self.encoder.forward(
            sequence_tokens=sequence_tokens,
            sequence_id=sequence_id,
        )

        # Grab the classification token <CLS> embeddings.
        x = out.embeddings[:, 0, :]

        z = self.cc_head.forward(x)

        return z

    def forward_all(
        self, sequence_tokens: Tensor, sequence_id: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        out: ESMCOutput = self.encoder.forward(
            sequence_tokens=sequence_tokens,
            sequence_id=sequence_id,
        )

        # Grab the classification token <CLS> embeddings.
        x = out.embeddings[:, 0, :]

        z_mf = self.mf_head.forward(x)
        z_bp = self.bp_head.forward(x)
        z_cc = self.cc_head.forward(x)

        return z_mf, z_bp, z_cc

    @torch.inference_mode()
    def predict_terms(
        self, sequence_tokens: Tensor, top_p: float = 0.5
    ) -> tuple[dict[str, float], ...]:
        """Predicts GO terms based on the input sequence tokens."""

        assert sequence_tokens.ndim == 1, "sequence must be a 1D tensor."
        assert 0 < top_p <= 1, "top_p must be in the range (0, 1]."

        z_mf, z_bp, z_cc = self.forward_all(sequence_tokens.unsqueeze(0))

        mf_prob = torch.sigmoid(z_mf).squeeze(0).tolist()
        bp_prob = torch.sigmoid(z_bp).squeeze(0).tolist()
        cc_prob = torch.sigmoid(z_cc).squeeze(0).tolist()

        aspects = [
            (self.indexToMfGoTerm, mf_prob),
            (self.indexToBpGoTerm, bp_prob),
            (self.indexToCcGoTerm, cc_prob),
        ]

        for mapping, probabilities in aspects:
            probabilities = {
                mapping[index]: probability
                for index, probability in enumerate(copy(probabilities))
                if probability > top_p
            }

        return mf_prob, bp_prob, cc_prob

    @torch.inference_mode()
    def predict_subgraph(
        self, sequence_tokens: Tensor, top_p: float = 0.5
    ) -> tuple[DiGraph, ...]:
        """Predicts a subgraph of the GO based on the input sequence tokens."""

        assert self.graph is not None, "Gene Ontology graph is not loaded."

        mf_prob, bp_prob, cc_prob = self.predict_terms(sequence_tokens, top_p)

        mf_subgraph, bp_subgraph, cc_subgraph = None, None, None

        aspects = [
            (mf_prob, mf_subgraph),
            (bp_prob, bp_subgraph),
            (cc_prob, cc_subgraph),
        ]

        for probabilities, subgraph in aspects:
            child_nodes = copy(probabilities)

            probabilities = defaultdict(float, probabilities)

            # Fix up the predictions by leveraging the GO DAG hierarchy.
            for go_id, child_probability in child_nodes.items():
                for descendant in descendants(self.graph, go_id):
                    parent_probability = probabilities[descendant]

                    probabilities[descendant] = max(
                        parent_probability,
                        child_probability,
                    )

            subgraph = self.graph.subgraph(probabilities.keys())

        return mf_subgraph, bp_subgraph, cc_subgraph


class MultiLabelClassifier(Module):
    """A 2-layer multi-label binary classification head with SwiGLU activation."""

    def __init__(self, embedding_dimensions: int, num_classes: int):
        super().__init__()

        assert embedding_dimensions > 0, "embedding_dimensions must be greater than 0."
        assert num_classes > 0, "num_classes must be greater than 0."

        self.linear1 = Linear(embedding_dimensions, 2 * embedding_dimensions)
        self.linear2 = Linear(embedding_dimensions, num_classes)

        self.swiglu = SwiGLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1(x)
        z = self.swiglu(z)
        z = self.linear2(z)

        return z
