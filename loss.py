import torch

from torch import Tensor

from torch.nn import Module, Parameter


class AdaptiveAspectWeighting(Module):
    """
    Adaptively weighting the loss of each aspect (MF, BP, CC) in the Gene Ontology prediction task.
    """

    def __init__(self, aspects: set, min_weight: float):
        super().__init__()

        assert len(aspects) > 0, "Aspects list must contain at least 1 aspect."
        assert min_weight > 0.0, "Minimum weight must be greater than 0."

        aspect_mapping = {aspect: i for i, aspect in enumerate(aspects)}

        self.log_sigmas = Parameter(torch.zeros(len(aspects)))

        self.aspect_mapping = aspect_mapping
        self.min_weight = min_weight

    @property
    def weights(self) -> list[float]:
        """
        Get current loss weights based on learned uncertainties.

        Returns:
            List of loss weights for each task.
        """

        weights = torch.exp(-2.0 * self.log_sigmas)

        weights = weights.clamp(min=self.min_weight)

        weights = weights.detach().tolist()

        return weights

    def forward(self, loss: Tensor, aspect: str) -> Tensor:
        assert (
            aspect in self.aspect_mapping
        ), f"Invalid aspect '{aspect}' given, must be one of {set(self.aspect_mapping.keys())}."

        index = self.aspect_mapping[aspect]

        log_sigma = self.log_sigmas[index]

        weight = torch.exp(-2.0 * log_sigma)

        weight = weight.clamp(min=self.min_weight)

        loss = 0.5 * weight * loss + log_sigma

        return loss
