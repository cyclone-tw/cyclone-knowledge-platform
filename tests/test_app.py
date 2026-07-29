"""The HTTP surface, and whether config actually reaches it.

An endpoint that reports four fields is only worth having if the fields track
reality. The last test here is the one that matters: change the bundle through
config, and the reported revision must change with it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ckp.app import create_app
from ckp.config import load_config
from ckp.revision import API_VERSION, compute_index_revision
from conftest import REPO_ROOT

FIXTURE_BUNDLE = REPO_ROOT / "fixtures" / "synthetic-bundle"
CONTRACT_FIELDS = ("profile_version", "api_version", "bundle_commit", "index_revision")


@pytest.fixture
def client() -> TestClient:
    config = load_config(env={"CKP_BUNDLE_ROOT": str(FIXTURE_BUNDLE)})
    return TestClient(create_app(config))


def test_health_is_ok_with_a_readable_bundle(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"config_loaded": True, "bundle_readable": True}
    assert body["config_layers"][0] == "defaults"


def test_health_is_degraded_when_the_bundle_is_missing(tmp_path) -> None:
    """Up but unable to serve is a real state, and must not read as healthy."""
    config = load_config(env={"CKP_BUNDLE_ROOT": str(tmp_path / "absent")})
    response = TestClient(create_app(config)).get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["bundle_readable"] is False


def test_revision_returns_all_four_contract_fields(client: TestClient) -> None:
    response = client.get("/revision")
    assert response.status_code == 200
    body = response.json()
    for field in CONTRACT_FIELDS:
        assert field in body, f"contract field {field} missing from /revision"
    assert body["api_version"] == API_VERSION
    assert body["profile_version"] == "cyclone-profile-v1"
    assert body["index_revision"].startswith("sha256:")
    assert body["bundle_commit"]


def test_revision_names_the_evidence_for_each_field(client: TestClient) -> None:
    sources = client.get("/revision").json()["sources"]
    assert set(sources) == set(CONTRACT_FIELDS)
    assert sources["api_version"] == "code"
    assert sources["index_revision"] == "computed"


def test_api_version_has_a_single_source(client: TestClient) -> None:
    """The endpoint, the app, and the published schema must not drift apart."""
    app_version = client.app.version
    schema_version = client.get("/openapi.json").json()["info"]["version"]
    endpoint_version = client.get("/revision").json()["api_version"]
    assert app_version == schema_version == endpoint_version == API_VERSION


def test_index_revision_matches_the_locally_computed_digest(client: TestClient) -> None:
    """The endpoint reports the derived value, not a stored copy of one."""
    served = client.get("/revision").json()["index_revision"]
    assert served == compute_index_revision(FIXTURE_BUNDLE, "**/*.md")


def test_config_override_reaches_the_endpoint(tmp_path) -> None:
    """Config layering is only real if it changes what the service reports."""
    alternate = tmp_path / "other-bundle"
    alternate.mkdir()
    (alternate / "only.md").write_bytes(b"different bundle\n")
    (alternate / "bundle.toml").write_text(
        'profile_version = "other-profile"\n', encoding="utf-8"
    )

    default_body = (
        TestClient(
            create_app(load_config(env={"CKP_BUNDLE_ROOT": str(FIXTURE_BUNDLE)}))
        )
        .get("/revision")
        .json()
    )
    overridden_body = (
        TestClient(create_app(load_config(env={"CKP_BUNDLE_ROOT": str(alternate)})))
        .get("/revision")
        .json()
    )

    assert overridden_body["profile_version"] == "other-profile"
    assert overridden_body["index_revision"] != default_body["index_revision"]
    assert overridden_body["index_revision"] == compute_index_revision(
        alternate, "**/*.md"
    )
