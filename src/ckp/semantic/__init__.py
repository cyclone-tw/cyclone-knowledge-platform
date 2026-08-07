"""Offline semantic embedding provider (issue #25 / Epic #21 D4).

C4 (``ckp.embedding``) stays frozen: this package is a second consumer of
its provider-neutral interfaces, the same relationship C5's index package
already has (``tests/test_c4_contract.py`` allows both). Nothing in
``ckp.embedding`` changes to make this package exist.

Real BGE-family sentence embeddings run fully offline at query time --
``ckp.semantic.provider`` never imports anything network-capable. Model
weights are pinned by content digest in ``ckp.semantic.manifest`` and
fetched only at install/build time by ``scripts/fetch-semantic-model.sh``
(D4 amendment: build-time fetch of pinned weights is allowed, query-time
network never is).

The inference dependencies (``onnxruntime``, ``tokenizers``, ``numpy``) live
behind the ``semantic`` extra and stay out of the base dependencies and the
runtime image, exactly like ``qdrant-client`` for C5 -- importing this
package does not require them; only constructing
:class:`~ckp.semantic.provider.SemanticEmbeddingProvider` does.

Builders only, as in ``ckp.embedding`` and the C5 index package: no module-level
registry, no default model directory, no default provider name.
"""

from ckp.semantic.assets import SemanticModelAssets, resolve_semantic_assets
from ckp.semantic.composition import (
    SemanticEmbeddingStack,
    build_semantic_registry,
    build_semantic_stack,
)
from ckp.semantic.errors import (
    SEMANTIC_ERROR_PREFIX,
    SemanticErrorCode,
    SemanticRefusal,
)
from ckp.semantic.manifest import (
    MODEL_REPO_ID,
    MODEL_REVISION,
    SEMANTIC_MODEL_DIMENSION,
    SEMANTIC_MODEL_FILES,
    SEMANTIC_MODEL_MAX_SEQUENCE_LENGTH,
    SemanticAssetFile,
)
from ckp.semantic.provider import (
    PROVIDER_IMPLEMENTATION_VERSION,
    SEMANTIC_EMBEDDING_ID,
    SemanticEmbeddingProvider,
)

__all__ = [
    "MODEL_REPO_ID",
    "MODEL_REVISION",
    "PROVIDER_IMPLEMENTATION_VERSION",
    "SEMANTIC_ERROR_PREFIX",
    "SEMANTIC_EMBEDDING_ID",
    "SEMANTIC_MODEL_DIMENSION",
    "SEMANTIC_MODEL_FILES",
    "SEMANTIC_MODEL_MAX_SEQUENCE_LENGTH",
    "SemanticAssetFile",
    "SemanticEmbeddingProvider",
    "SemanticEmbeddingStack",
    "SemanticErrorCode",
    "SemanticModelAssets",
    "SemanticRefusal",
    "build_semantic_registry",
    "build_semantic_stack",
    "resolve_semantic_assets",
]
