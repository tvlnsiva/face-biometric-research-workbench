"""Abstract recognition embedder contract and L2 normalization utilities (Milestone P1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class MissingModelArtifactError(FileNotFoundError):
    """Raised when a required recognition model weight/ONNX artifact is not present locally."""


def l2_normalize_embeddings(raw_embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 2D `(N, D)` embedding matrix row-wise with zero-norm protection."""
    arr = np.asarray(raw_embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix (N, D); got shape={arr.shape}")
    if arr.shape[0] == 0:
        return arr.copy()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, eps)
    normalized = arr / safe_norms
    # Zero rows remain zero rather than amplifying noise
    zero_mask = (norms.squeeze(-1) <= eps)
    if np.any(zero_mask):
        normalized[zero_mask] = 0.0
    return normalized.astype(np.float32, copy=False)


@dataclass(frozen=True)
class EmbeddingBatchOutput:
    """Structured output from batched facial feature extraction."""

    raw_embeddings: np.ndarray
    embeddings: np.ndarray
    l2_normalized: bool
    embedding_dim: int
    model_id: str
    model_version: str
    runtime: str
    device: str
    precision: str
    is_synthetic_fixture: bool
    preprocess_time_ms: float
    inference_time_ms: float
    postprocess_time_ms: float


class FaceEmbedder(ABC):
    """Abstract interface for facial feature extractors."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Canonical model identifier."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model version or commit tag."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output feature vector dimensionality."""

    @property
    @abstractmethod
    def is_synthetic_fixture(self) -> bool:
        """True ONLY for synthetic test fixtures; False for real biometric models."""

    @property
    @abstractmethod
    def initialization_time_ms(self) -> float:
        """One-time model load latency in milliseconds."""

    @abstractmethod
    def preprocess(self, aligned_crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Convert aligned uint8 BGR crops into normalized NCHW tensors."""

    @abstractmethod
    def infer_preprocessed(self, batch_nchw: np.ndarray) -> np.ndarray:
        """Run forward pass on preprocessed NCHW batch and return raw float32 embeddings."""

    @abstractmethod
    def extract(
        self,
        aligned_crops_bgr: list[np.ndarray],
        l2_normalize: bool | None = None,
    ) -> EmbeddingBatchOutput:
        """Execute separated preprocess -> infer -> optional L2 normalization."""
