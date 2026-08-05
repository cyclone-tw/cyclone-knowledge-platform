"""C4 scope, offline-boundary, and composition tripwires."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from fastapi.testclient import TestClient

import ckp.embedding as embedding_package
from ckp.app import create_app
from ckp.config import load_config
from ckp.embedding.composition import (
    build_c4_deterministic_registry,
    build_c4_deterministic_stack,
)
from ckp.embedding.hashing import CosineReranker, HashEmbeddingProvider
from ckp.embedding.provider import EmbeddingProvider, RerankerProvider
from ckp.embedding.registry import ProviderRegistry
from conftest import REPO_ROOT

PACKAGE = REPO_ROOT / "src/ckp/embedding"

#: Everything C4 is allowed to import. The Epic puts C4 on C1 alone, and the
#: package is pure computation: no config, no bundle, no HTTP, no Git. An
#: allowlist over the AST catches what a substring scan of the source cannot,
#: because a docstring that merely says "no network" does not trip it.
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "hashlib",
        "math",
        "pydantic",
        "re",
        "typing",
        # Normalization only. Unicode's stability policy pins the normalized
        # form of an already-assigned string across versions, which is why
        # this one table is safe to consult and ``\\w``/``casefold`` are not.
        "unicodedata",
    }
)


def _module_paths() -> list[Path]:
    return sorted(PACKAGE.glob("*.py"))


def _imported_modules() -> set[str]:
    imported: set[str] = set()
    for path in _module_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise AssertionError(f"relative import in {path.name}")
                imported.add(node.module or "")
    return imported


def test_c4_expected_embedding_modules_are_present() -> None:
    assert {path.name for path in _module_paths()} == {
        "__init__.py",
        "composition.py",
        "errors.py",
        "hashing.py",
        "models.py",
        "provider.py",
        "registry.py",
    }


def test_the_package_imports_nothing_that_could_reach_a_network_or_a_secret() -> None:
    """No HTTP client, no cloud SDK, no ``os``, no ``subprocess``, no model loader.

    This is the C4 non-goal made mechanical: a cloud provider cannot be added
    without this test going red, which forces the decision through an issue.
    """
    for module in _imported_modules():
        root = module.split(".")[0]
        if root == "ckp":
            assert module.startswith("ckp.embedding"), module
            continue
        assert root in ALLOWED_IMPORT_ROOTS, module


def test_the_package_opens_no_file_and_evaluates_no_code() -> None:
    forbidden = {"open", "eval", "exec", "compile", "__import__", "input"}
    for path in _module_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden, f"{path.name}: {node.func.id}"


def test_the_package_exports_no_prebuilt_provider_or_registry() -> None:
    """Builders only. A module-level instance is a default nobody chose."""
    for name in embedding_package.__all__:
        value = getattr(embedding_package, name)
        assert not isinstance(value, ProviderRegistry), name
        if isinstance(value, type):
            continue
        assert not isinstance(value, EmbeddingProvider), name
        assert not isinstance(value, RerankerProvider), name


def test_the_c4_builders_have_no_permissive_defaults() -> None:
    expected = {"dimension", "embedding_name", "reranker_name"}
    for builder in (build_c4_deterministic_registry, build_c4_deterministic_stack):
        signature = inspect.signature(builder)
        assert set(signature.parameters) == expected, builder.__name__
        for parameter in signature.parameters.values():
            assert parameter.default is inspect.Parameter.empty, parameter.name
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, parameter.name

    for constructor in (HashEmbeddingProvider.__init__, CosineReranker.__init__):
        for parameter in inspect.signature(constructor).parameters.values():
            if parameter.name == "self":
                continue
            assert parameter.default is inspect.Parameter.empty, parameter.name
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, parameter.name


def test_default_app_composes_no_embedding_and_exposes_no_embedding_surface() -> None:
    bundle = REPO_ROOT / "fixtures/synthetic-bundle"
    app = create_app(load_config(env={"CKP_BUNDLE_ROOT": str(bundle)}))
    client = TestClient(app)
    paths = set(client.get("/openapi.json").json()["paths"])

    assert "embedding" not in vars(app.state)
    assert "reranker" not in vars(app.state)
    assert not any(
        token in path.lower()
        for path in paths
        for token in ("embed", "rerank", "vector", "index")
    )


def test_no_other_package_depends_on_c4_yet() -> None:
    """C5 wires the index. Until then this package is additive and revertible."""
    for path in sorted((REPO_ROOT / "src/ckp").rglob("*.py")):
        if PACKAGE in path.parents:
            continue
        assert "ckp.embedding" not in path.read_text(encoding="utf-8"), path.name


def test_the_shipped_providers_declare_themselves_offline_and_non_semantic() -> None:
    embedder = HashEmbeddingProvider(dimension=8)
    for descriptor in (
        embedder.descriptor,
        CosineReranker(embedder=embedder).descriptor,
    ):
        assert descriptor.requires_network is False
        assert descriptor.deterministic is True
        assert descriptor.semantic is False
