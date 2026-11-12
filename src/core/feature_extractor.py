"""
Feature Extractor — MobileNetV3-Large backbone for label embedding.
Produces L2-normalised float32 embeddings used for template matching.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Wraps MobileNetV3-Large (pretrained ImageNet) as a fixed feature extractor.
    The classifier head is removed; the remaining convolutional stack produces
    a spatial feature map that is adaptively average-pooled and flattened into
    a 1-D embedding, then L2-normalised.
    """

    def __init__(self, config: dict) -> None:
        self.embedding_dim: int = int(config.get("embedding_dim", 512))
        self._model = None
        self._transform = None
        self._device: Optional[str] = None  # Resolved at load time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        if self._device is None:
            import torch  # type: ignore
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load_model(self) -> None:
        """Lazy-load MobileNetV3-Large and build the preprocessing pipeline."""
        import torch  # type: ignore
        import torchvision.models as models  # type: ignore
        import torchvision.transforms as T  # type: ignore

        logger.info("Loading MobileNetV3-Large backbone on device '%s'.", self.device)

        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        # Keep everything except the final classifier (children[-1])
        # backbone.children() → [features, avgpool, classifier]
        feature_layers = list(backbone.children())[:-1]  # features + avgpool
        self._model = torch.nn.Sequential(*feature_layers)
        self._model.eval().to(self.device)

        self._transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.info("FeatureExtractor loaded (embedding_dim=%d).", self.embedding_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract an L2-normalised embedding from *image* (BGR numpy array).

        Parameters
        ----------
        image:
            H×W×3 uint8 BGR numpy array (OpenCV convention).

        Returns
        -------
        np.ndarray
            1-D float32 array of shape (embedding_dim,), L2-normalised.
        """
        import torch  # type: ignore

        if self._model is None:
            self._load_model()

        # BGR → RGB
        rgb = image[..., ::-1].copy()

        tensor = self._transform(rgb)  # (3, 224, 224)
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            features = self._model(tensor)  # (1, C, h, w) or (1, C, 1, 1) after avgpool

        # Flatten to 1-D vector
        vec: np.ndarray = features.squeeze().cpu().numpy().flatten().astype(np.float32)

        # Adapt dimension: truncate or zero-pad to embedding_dim
        if vec.shape[0] > self.embedding_dim:
            vec = vec[: self.embedding_dim]
        elif vec.shape[0] < self.embedding_dim:
            vec = np.pad(vec, (0, self.embedding_dim - vec.shape[0]))

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        return vec

    def warmup(self, rounds: int = 1) -> None:
        """Pre-load GPU kernels by running dummy inference."""
        if self._model is None:
            self._load_model()

        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(rounds):
            self.extract(dummy)

        logger.info("FeatureExtractor warmup complete (%d round(s)).", rounds)
