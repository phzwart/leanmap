"""leanmap: a small, self-contained, deployable parametric UMAP.

Build a fuzzy topological graph from high-dimensional data with FAISS k-NN
(no umap-learn dependency), train a PCA-anchored neural network to embed it,
then ``transform`` any new data through the saved model.
"""

from __future__ import annotations

from ._api import MapperConfig, LeanMap
from ._graph import (
    FuzzyGraphData,
    build_fuzzy_graph,
    faiss_knn,
    fit_ab_params,
    fuzzy_graph_from_knn,
    smooth_knn_dist,
    standardize,
)
from ._inducing import (
    coverage_radius,
    farthest_point_sampling,
    induce_embed,
    select_landmarks,
)
from ._decoder import CondFlow, GenerativeDecoder, MeanDecoder
from ._discriminator import LeanmapDiscriminator
from ._model import AttentionMapper, DeployableMapper, ParametricMapper, pca_components
from ._pipeline import pca_reduce, run_pipeline
from ._train import train_attention_mapper, train_parametric_mapper, transform

__version__ = "0.1.0"

__all__ = [
    "LeanMap",
    "MapperConfig",
    "FuzzyGraphData",
    "build_fuzzy_graph",
    "faiss_knn",
    "fuzzy_graph_from_knn",
    "smooth_knn_dist",
    "standardize",
    "fit_ab_params",
    "ParametricMapper",
    "DeployableMapper",
    "pca_components",
    "train_parametric_mapper",
    "train_attention_mapper",
    "AttentionMapper",
    "transform",
    "select_landmarks",
    "farthest_point_sampling",
    "induce_embed",
    "coverage_radius",
    "GenerativeDecoder",
    "MeanDecoder",
    "CondFlow",
    "LeanmapDiscriminator",
    "run_pipeline",
    "pca_reduce",
    "__version__",
]
