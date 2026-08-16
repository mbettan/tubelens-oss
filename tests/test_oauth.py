"""
Unit and Integration Tests for OAuth 2.0 Authorization Server & Dynamic Client Registration
"""

import base64
import hashlib

import pytest
from starlette.testclient import TestClient

from src.oauth import oauth_server
from src.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_oauth_metadata_discovery(client: TestClient) -> None:
    """Test RFC 8414 OAuth 2.0 Authorization Server Metadata."""
    r = client.get("/.well-known/oauth-authorization-server")
    assert r.status_code == 200
    data = r.json()
    assert "issuer" in data
    assert data["authorization_endpoint"].endswith("/oauth/authorize")
    assert data["token_endpoint"].endswith("/oauth/token")
    assert data["registration_endpoint"].endswith("/oauth/register")
    assert "authorization_code" in data["grant_types_supported"]
    assert "S256" in data["code_challenge_methods_supported"]

    # Also check OpenID configuration alias
    r_oidc = client.get("/.well-known/openid-configuration")
    assert r_oidc.status_code == 200
    assert r_oidc.json()["authorization_endpoint"] == data["authorization_endpoint"]


def test_dynamic_client_registration(client: TestClient) -> None:
    """Test RFC 7591 Dynamic Client Registration."""
    payload = {
        "client_name": "Claude Desktop Integration",
        "redirect_uris": ["https://claude.ai/api/oauth/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    r = client.post("/oauth/register", json=payload)
    assert r.status_code == 201
    data = r.json()
    assert data["client_id"].startswith("tubelens_")
    assert len(data["client_secret"]) > 16
    assert data["client_name"] == "Claude Desktop Integration"
    assert data["redirect_uris"] == ["https://claude.ai/api/oauth/callback"]


def test_oauth_authorization_code_flow_with_pkce(client: TestClient) -> None:
    """Test full OAuth 2.0 Authorization Code Flow with PKCE (RFC 7636)."""
    # 1. Register dynamic client
    reg_res = client.post(
        "/oauth/register",
        json={"client_name": "Claude Test", "redirect_uris": ["https://claude.ai/callback"]},
    )
    client_info = reg_res.json()
    client_id = client_info["client_id"]
    client_secret = client_info["client_secret"]
    redirect_uri = "https://claude.ai/callback"

    # 2. PKCE Setup
    code_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    # 3. GET /oauth/authorize renders consent UI
    r_auth_get = client.get(
        "/oauth/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": "state_xyz_123",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        },
    )
    assert r_auth_get.status_code == 200
    assert "Connect Claude Test" in r_auth_get.text
    assert "TubeLens OSS" in r_auth_get.text

    # 4. POST /oauth/authorize submits consent approval
    r_auth_post = client.post(
        "/oauth/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": "state_xyz_123",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        },
        data={
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": "state_xyz_123",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        },
        follow_redirects=False,
    )
    assert r_auth_post.status_code == 302
    location = r_auth_post.headers["location"]
    assert location.startswith("https://claude.ai/callback?")
    assert "state=state_xyz_123" in location
    assert "code=tl_code_" in location

    # Extract code
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(location)
    query_params = parse_qs(parsed.query)
    auth_code = query_params["code"][0]

    # 5. POST /oauth/token exchange with code & PKCE verifier
    token_res = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
            "code_verifier": code_verifier,
        },
    )
    assert token_res.status_code == 200
    token_data = token_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "Bearer"
    assert "refresh_token" in token_data
    access_token = token_data["access_token"]
    refresh_token = token_data["refresh_token"]

    # 6. GET /oauth/userinfo with Bearer token
    userinfo_res = client.get(
        "/oauth/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert userinfo_res.status_code == 200
    assert userinfo_res.json()["sub"] == client_id

    # 7. POST /oauth/token refresh grant
    ref_res = client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
    )
    assert ref_res.status_code == 200
    new_tokens = ref_res.json()
    assert "access_token" in new_tokens
    assert new_tokens["access_token"] != access_token


def test_oauth_pkce_failure_on_bad_verifier(client: TestClient) -> None:
    """Test PKCE rejection when code_verifier is invalid."""
    code_verifier = "correct_verifier_secret_12345678901234567890"
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    # Create code directly
    code = oauth_server.create_authorization_code(
        client_id="claude-desktop",
        redirect_uri="https://claude.ai/callback",
        code_challenge=code_challenge,
        code_challenge_method="S256",
    )

    # Attempt exchange with wrong verifier
    token_res = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": "https://claude.ai/callback",
            "client_id": "claude-desktop",
            "code_verifier": "wrong_verifier_here",
        },
    )
    assert token_res.status_code == 400
    assert "PKCE verification failed" in token_res.json()["error_description"]


def test_oauth_admin_password_protection(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that oauth_admin_password enforces password entry on /oauth/authorize."""
    from src.config import settings

    monkeypatch.setattr(settings, "oauth_admin_password", "super-secret-admin-pass")

    client_id = "claude-desktop"
    redirect_uri = "https://claude.ai/api/oauth/callback"

    # 1. GET /oauth/authorize displays admin password input
    r_get = client.get(
        "/oauth/authorize",
        params={"client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code"},
    )
    assert r_get.status_code == 200
    assert 'name="admin_password"' in r_get.text
    assert "Admin Password" in r_get.text

    # 2. POST /oauth/authorize with invalid password fails and stays on page
    r_post_invalid = client.post(
        "/oauth/authorize",
        params={"client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code"},
        data={
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "admin_password": "wrong-password",
        },
        follow_redirects=False,
    )
    assert r_post_invalid.status_code == 200
    assert "Invalid admin password" in r_post_invalid.text

    # 3. POST /oauth/authorize with correct password succeeds and redirects
    r_post_valid = client.post(
        "/oauth/authorize",
        params={"client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code"},
        data={
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "admin_password": "super-secret-admin-pass",
        },
        follow_redirects=False,
    )
    assert r_post_valid.status_code == 302
    assert r_post_valid.headers["location"].startswith(redirect_uri)
    assert "code=tl_code_" in r_post_valid.headers["location"]


def test_oauth_registration_token_protection(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that oauth_registration_token gates dynamic client registration."""
    from src.config import settings

    monkeypatch.setattr(settings, "oauth_registration_token", "reg-secret-token-123")

    payload = {
        "client_name": "Protected Client",
        "redirect_uris": ["https://example.com/callback"],
    }

    # 1. Registration without token fails
    r_unauth = client.post("/oauth/register", json=payload)
    assert r_unauth.status_code == 401
    assert r_unauth.json()["error"] == "unauthorized"

    # 2. Registration with invalid token fails
    r_bad = client.post(
        "/oauth/register",
        json=payload,
        headers={"Authorization": "Bearer bad-token"},
    )
    assert r_bad.status_code == 401

    # 3. Registration with valid token succeeds
    r_ok = client.post(
        "/oauth/register",
        json=payload,
        headers={"Authorization": "Bearer reg-secret-token-123"},
    )
    assert r_ok.status_code == 201
    assert r_ok.json()["client_name"] == "Protected Client"

