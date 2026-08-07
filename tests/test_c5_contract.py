"""C5 scope and composition tripwires."""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from ckp.app import create_app
from ckp.config import load_config
from ckp.index.models import plan_rebuild
from ckp.index.qdrant import QdrantVectorIndex
from conftest import REPO_ROOT
from index_fixtures import UNUSED_ROOT  # noqa: F401 - shared fixture import


def test_default_app_exposes_no_index_surface() -> None:
    bundle = REPO_ROOT / "fixtures/synthetic-bundle"
    app = create_app(load_config(env={"CKP_BUNDLE_ROOT": str(bundle)}))
    client = TestClient(app)
    paths = set(client.get("/openapi.json").json()["paths"])

    assert "index" not in vars(app.state)
    assert not any(
        token in path.lower()
        for path in paths
        for token in ("vector", "embed", "rerank", "snapshot", "benchmark")
    )


def test_no_module_outside_the_index_package_imports_it() -> None:
    for path in sorted((REPO_ROOT / "src/ckp").rglob("*.py")):
        if path.is_relative_to(REPO_ROOT / "src/ckp/index"):
            continue
        source = path.read_text(encoding="utf-8")
        assert "ckp.index" not in source, path


def test_c5_expected_index_modules_are_present() -> None:
    module_names = {path.name for path in (REPO_ROOT / "src/ckp/index").glob("*.py")}
    assert module_names == {
        "__init__.py",
        "errors.py",
        "memory.py",
        "models.py",
        "provider.py",
        "qdrant.py",
        "revision.py",
    }
    benchmark_names = {path.name for path in (REPO_ROOT / "benchmarks").glob("*.py")}
    assert benchmark_names == {
        "__init__.py",
        "questions.py",
        "export_compare.py",
        "resilience.py",
        "shadow.py",
    }


def test_index_package_stays_inside_the_offline_boundary() -> None:
    """No cloud client, credential read, subprocess, or ad-hoc file IO."""
    for path in sorted((REPO_ROOT / "src/ckp/index").glob("*.py")):
        source = path.read_text(encoding="utf-8").lower()
        for forbidden in (
            "httpx",
            "requests",
            "urllib",
            "socket",
            "openai",
            "cohere",
            "voyage",
            "torch",
            "sentence_transformers",
            "os.environ",
            "subprocess",
            "api_key",
            "open(",
        ):
            assert forbidden not in source, (path, forbidden)


def test_abstraction_modules_name_no_concrete_provider() -> None:
    """Only ``qdrant.py``/``memory.py`` (and the exporter) know the names."""
    for module in ("models.py", "provider.py", "errors.py", "revision.py"):
        source = (REPO_ROOT / "src/ckp/index" / module).read_text(encoding="utf-8")
        for provider_token in ("qdrant", "memory-cosine", "third-party"):
            assert provider_token not in source.lower(), (module, provider_token)


def test_qdrant_module_is_the_only_place_that_imports_the_client() -> None:
    """Prose may mention Qdrant; only ``index/qdrant.py`` may import it."""
    for path in sorted((REPO_ROOT / "src/ckp").rglob("*.py")):
        if path.name == "qdrant.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert "qdrant_client" not in source, path


def test_qdrant_client_is_not_a_base_dependency() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    base = pyproject.split("[project.optional-dependencies]")[0]
    assert "qdrant" not in base.lower()
    assert 'index = [\n  "qdrant-client' in pyproject


def test_composition_has_no_permissive_defaults() -> None:
    for target in (plan_rebuild,):
        signature = inspect.signature(target)
        assert all(
            parameter.default is inspect.Parameter.empty
            and parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        ), target
    qdrant_signature = inspect.signature(QdrantVectorIndex.__init__)
    assert all(
        parameter.default is inspect.Parameter.empty
        for name, parameter in qdrant_signature.parameters.items()
        if name != "self"
    )


def test_error_codes_are_prefixed_and_unique() -> None:
    from ckp.index.errors import INDEX_ERROR_PREFIX, IndexErrorCode

    values = [code.value for code in IndexErrorCode]
    assert len(values) == len(set(values))
    assert all(value.startswith(f"{INDEX_ERROR_PREFIX}/") for value in values)


def test_composed_revision_source_covers_every_input() -> None:
    from ckp.index.revision import compute_composed_index_revision

    source = inspect.getsource(compute_composed_index_revision)
    for needle in (
        "REVISION_DOMAIN",
        "bundle_index_revision",
        "embedding_revision",
        "index_schema_version",
    ):
        assert needle in source


def test_benchmark_corpus_is_synthetic_and_flagged() -> None:
    from benchmarks.questions import CORPUS

    for note in CORPUS:
        assert "Synthetic" in note.body or "synthetic" in note.body, note.relative_path
