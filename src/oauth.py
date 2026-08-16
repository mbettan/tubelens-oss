"""
OAuth 2.0 Authorization Server & Dynamic Client Registration (RFC 6749, RFC 7591, RFC 7636, RFC 8414)
Enables Anthropic Claude.ai, Cursor, ChatGPT, and AI agents to connect via standard OAuth 2.0 authorization code flow.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlparse

import jwt
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from src.config import settings

logger = logging.getLogger("oauth_server")


@dataclass
class RegisteredClient:
    """OAuth 2.0 registered client metadata (RFC 7591)."""

    client_id: str
    client_secret: str
    client_name: str
    redirect_uris: list[str]
    grant_types: list[str] = field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = field(default_factory=lambda: ["code"])
    token_endpoint_auth_method: str = "client_secret_post"
    created_at: float = field(default_factory=time.time)


@dataclass
class AuthorizationCode:
    """Pending authorization code awaiting token exchange."""

    code: str
    client_id: str
    redirect_uri: str
    scope: str
    code_challenge: str | None = None
    code_challenge_method: str | None = None
    state: str | None = None
    expires_at: float = field(default_factory=lambda: time.time() + 600)  # 10 min TTL


class OAuthServer:
    """In-memory RFC-compliant OAuth 2.0 authorization server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[str, RegisteredClient] = {}
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._refresh_tokens: dict[str, dict[str, Any]] = {}
        self._secret = settings.oauth_jwt_secret or secrets.token_hex(32)

        # Pre-seed a default client for standard configurations
        default_client = RegisteredClient(
            client_id="claude-desktop",
            client_secret=secrets.token_urlsafe(32),
            client_name="Claude Desktop / Claude.ai",
            redirect_uris=["https://claude.ai/api/oauth/callback", "http://localhost:8080/callback"],
        )
        self._clients[default_client.client_id] = default_client

    def get_issuer_url(self, request: Request) -> str:
        """Derive canonical base issuer URL from settings or incoming request headers."""
        if settings.oauth_issuer_url:
            return settings.oauth_issuer_url.rstrip("/")

        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
        return f"{proto}://{host}"

    def register_client(
        self,
        client_name: str,
        redirect_uris: list[str],
        grant_types: list[str] | None = None,
        response_types: list[str] | None = None,
        token_endpoint_auth_method: str = "client_secret_post",
    ) -> RegisteredClient:
        """Register a new dynamic OAuth client (RFC 7591)."""
        client_id = f"tubelens_{secrets.token_hex(16)}"
        client_secret = secrets.token_urlsafe(32)
        client = RegisteredClient(
            client_id=client_id,
            client_secret=client_secret,
            client_name=client_name or "AI Assistant Client",
            redirect_uris=redirect_uris,
            grant_types=grant_types or ["authorization_code", "refresh_token"],
            response_types=response_types or ["code"],
            token_endpoint_auth_method=token_endpoint_auth_method,
        )
        with self._lock:
            self._clients[client_id] = client
        logger.info(f"Registered new OAuth client '{client.client_name}' (client_id={client_id})")
        return client

    def get_client(self, client_id: str) -> RegisteredClient | None:
        """Look up a client by ID."""
        with self._lock:
            return self._clients.get(client_id)

    def create_authorization_code(
        self,
        client_id: str,
        redirect_uri: str,
        scope: str = "tubelens",
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
        state: str | None = None,
    ) -> str:
        """Create a single-use authorization code with PKCE parameters."""
        code = f"tl_code_{secrets.token_urlsafe(32)}"
        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            state=state,
        )
        with self._lock:
            self._cleanup_expired()
            self._auth_codes[code] = auth_code
        return code

    def exchange_code(
        self,
        code: str,
        client_id: str | None,
        client_secret: str | None,
        redirect_uri: str | None,
        code_verifier: str | None,
        issuer: str,
    ) -> dict[str, Any]:
        """Exchange authorization code for access and refresh tokens (RFC 6749 & PKCE RFC 7636)."""
        with self._lock:
            auth_code = self._auth_codes.pop(code, None)

        if not auth_code or auth_code.expires_at < time.time():
            raise ValueError("invalid_grant: Authorization code is invalid or expired")

        # Verify client if client_id was provided during authorization
        if auth_code.client_id and client_id and auth_code.client_id != client_id:
            raise ValueError("invalid_client: Client ID mismatch")

        client = self.get_client(auth_code.client_id)
        if client and client_secret:
            if not hmac.compare_digest(client.client_secret, client_secret):
                raise ValueError("invalid_client: Invalid client secret")

        # Verify redirect_uri
        if redirect_uri and auth_code.redirect_uri:
            parsed_req = urlparse(redirect_uri)
            parsed_auth = urlparse(auth_code.redirect_uri)
            if (parsed_req.scheme, parsed_req.netloc, parsed_req.path) != (
                parsed_auth.scheme,
                parsed_auth.netloc,
                parsed_auth.path,
            ):
                raise ValueError("invalid_grant: redirect_uri mismatch")

        # Verify PKCE if code_challenge was used
        if auth_code.code_challenge:
            if not code_verifier:
                raise ValueError("invalid_request: code_verifier is required for PKCE")
            if not self._verify_pkce(code_verifier, auth_code.code_challenge, auth_code.code_challenge_method):
                raise ValueError("invalid_grant: PKCE verification failed")

        # Generate tokens
        now = int(time.time())
        ttl = settings.oauth_token_ttl_seconds
        payload = {
            "jti": f"jwt_{secrets.token_hex(16)}",
            "iss": issuer,
            "sub": auth_code.client_id or "claude-user",
            "aud": issuer,
            "scope": auth_code.scope or "tubelens",
            "iat": now,
            "exp": now + ttl,
        }
        access_token = jwt.encode(payload, self._secret, algorithm="HS256")
        refresh_token = f"tl_refresh_{secrets.token_urlsafe(32)}"

        with self._lock:
            self._refresh_tokens[refresh_token] = {
                "client_id": auth_code.client_id,
                "scope": auth_code.scope,
                "created_at": now,
            }

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": ttl,
            "refresh_token": refresh_token,
            "scope": auth_code.scope or "tubelens",
        }

    def refresh_access_token(self, refresh_token: str, issuer: str) -> dict[str, Any]:
        """Issue a new access token using a valid refresh token."""
        with self._lock:
            token_data = self._refresh_tokens.get(refresh_token)

        if not token_data:
            raise ValueError("invalid_grant: Invalid or expired refresh token")

        now = int(time.time())
        ttl = settings.oauth_token_ttl_seconds
        payload = {
            "jti": f"jwt_{secrets.token_hex(16)}",
            "iss": issuer,
            "sub": token_data.get("client_id", "claude-user"),
            "aud": issuer,
            "scope": token_data.get("scope", "tubelens"),
            "iat": now,
            "exp": now + ttl,
        }
        access_token = jwt.encode(payload, self._secret, algorithm="HS256")

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": ttl,
            "refresh_token": refresh_token,
            "scope": token_data.get("scope", "tubelens"),
        }

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify an HMAC-SHA256 JWT access token."""
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                options={"verify_aud": False},  # Allow local and proxy host variations
            )
            return payload
        except jwt.PyJWTError as e:
            logger.debug(f"JWT verification failed: {e}")
            return None

    def _verify_pkce(self, verifier: str, challenge: str, method: str | None) -> bool:
        """Verify RFC 7636 PKCE code_verifier against code_challenge."""
        if method == "plain" or not method:
            return hmac.compare_digest(verifier, challenge)
        if method == "S256":
            digest = hashlib.sha256(verifier.encode("ascii")).digest()
            calc_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
            return hmac.compare_digest(calc_challenge, challenge.rstrip("="))
        return False

    def _cleanup_expired(self) -> None:
        """Purge expired authorization codes."""
        now = time.time()
        self._auth_codes = {k: v for k, v in self._auth_codes.items() if v.expires_at > now}


oauth_server = OAuthServer()


# ------------------------------------------------------------------------------
# Starlette Route Handlers
# ------------------------------------------------------------------------------


async def handle_oauth_metadata(request: Request) -> JSONResponse:
    """
    RFC 8414 OAuth 2.0 Authorization Server Metadata & OpenID Connect Discovery.
    Claude and OAuth clients query this to discover all endpoints and capabilities.
    """
    issuer = oauth_server.get_issuer_url(request)
    metadata = {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/oauth/authorize",
        "token_endpoint": f"{issuer}/oauth/token",
        "registration_endpoint": f"{issuer}/oauth/register",
        "userinfo_endpoint": f"{issuer}/oauth/userinfo",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_post",
            "client_secret_basic",
            "none",
        ],
        "code_challenge_methods_supported": ["S256", "plain"],
        "scopes_supported": ["tubelens", "offline_access", "read"],
        "service_documentation": f"{issuer}/docs",
    }
    return JSONResponse(metadata)


async def handle_dynamic_client_registration(request: Request) -> Response:
    """
    RFC 7591 Dynamic Client Registration.
    Claude and automated MCP clients call this to register themselves dynamically.

    When OAUTH_REGISTRATION_TOKEN is set, callers must provide it as a
    Bearer token in the Authorization header to prevent open registration.
    """
    # Gate registration behind a secret token when configured
    required_token = (settings.oauth_registration_token or "").strip()
    if required_token:
        auth_header = request.headers.get("authorization", "")
        provided_token = ""
        if auth_header.lower().startswith("bearer "):
            provided_token = auth_header[7:].strip()
        if not provided_token or not secrets.compare_digest(provided_token, required_token):
            logger.warning("OAuth client registration rejected: missing or invalid registration token")
            return JSONResponse(
                {"error": "unauthorized", "error_description": "A valid registration token is required."},
                status_code=401,
            )

    try:
        data = await request.json()
    except Exception:
        data = {}

    client_name = data.get("client_name", "AI Assistant Client")
    redirect_uris = data.get("redirect_uris", [])
    grant_types = data.get("grant_types", ["authorization_code", "refresh_token"])
    response_types = data.get("response_types", ["code"])
    token_endpoint_auth_method = data.get("token_endpoint_auth_method", "client_secret_post")

    client = oauth_server.register_client(
        client_name=client_name,
        redirect_uris=redirect_uris,
        grant_types=grant_types,
        response_types=response_types,
        token_endpoint_auth_method=token_endpoint_auth_method,
    )

    issuer = oauth_server.get_issuer_url(request)
    response_payload = {
        "client_id": client.client_id,
        "client_secret": client.client_secret,
        "client_name": client.client_name,
        "redirect_uris": client.redirect_uris,
        "grant_types": client.grant_types,
        "response_types": client.response_types,
        "token_endpoint_auth_method": client.token_endpoint_auth_method,
        "client_id_issued_at": int(client.created_at),
        "client_secret_expires_at": 0,  # Never expires
        "registration_client_uri": f"{issuer}/oauth/register/{client.client_id}",
    }
    return JSONResponse(response_payload, status_code=201)


async def handle_oauth_authorize(request: Request) -> Response:
    """
    RFC 6749 Authorization Endpoint.
    Renders an approval page for user consent and issues authorization codes.
    """
    params = request.query_params
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    response_type = params.get("response_type", "code")
    state = params.get("state", "")
    scope = params.get("scope", "tubelens")
    code_challenge = params.get("code_challenge")
    code_challenge_method = params.get("code_challenge_method")

    if not redirect_uri:
        return Response("Missing redirect_uri parameter", status_code=400)

    if response_type != "code":
        error_params = {"error": "unsupported_response_type", "state": state}
        return RedirectResponse(f"{redirect_uri}?{urlencode(error_params)}", status_code=302)

    # Reject unknown clients — they must register first via /oauth/register
    client = oauth_server.get_client(client_id)
    if not client:
        logger.warning(f"OAuth authorize rejected: unknown client_id '{client_id}'")
        error_params = {"error": "unauthorized_client", "error_description": "Unknown client_id. Register via /oauth/register first.", "state": state}
        return RedirectResponse(f"{redirect_uri}?{urlencode(error_params)}", status_code=302)

    # Determine if admin password is required
    admin_password = (settings.oauth_admin_password or "").strip()
    password_required = bool(admin_password)
    auth_error_msg = ""

    # Handle POST approval submission
    if request.method == "POST":
        form_data = await request.form()

        # Verify admin password if configured
        if password_required:
            submitted_password = str(form_data.get("admin_password", "")).strip()
            if not submitted_password or not secrets.compare_digest(submitted_password, admin_password):
                logger.warning("OAuth authorize rejected: invalid admin password")
                auth_error_msg = "Invalid admin password. Please try again."
            # Fall through to re-render the consent page with error

        if not auth_error_msg:
            code = oauth_server.create_authorization_code(
                client_id=client_id,
                redirect_uri=redirect_uri,
                scope=scope,
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
                state=state,
            )
            cb_params = {"code": code}
            if state:
                cb_params["state"] = state
            sep = "&" if "?" in redirect_uri else "?"
            return RedirectResponse(f"{redirect_uri}{sep}{urlencode(cb_params)}", status_code=302)

    # Render branded consent HTML UI
    escaped_client = html.escape(client.client_name if client else "AI Assistant")
    escaped_redirect = html.escape(redirect_uri)
    escaped_state = html.escape(state)
    escaped_scope = html.escape(scope)
    escaped_challenge = html.escape(code_challenge or "")
    escaped_method = html.escape(code_challenge_method or "")
    escaped_error = html.escape(auth_error_msg)

    # Conditional password field HTML
    password_field_html = ""
    if password_required:
        error_html = (
            f'<div class="error-msg">{escaped_error}</div>'
            if escaped_error else ""
        )
        password_field_html = f"""
      <div class="password-section">
        {error_html}
        <label for="admin_password" class="pw-label">Admin Password</label>
        <input type="password" id="admin_password" name="admin_password"
               placeholder="Enter admin password to authorize"
               class="pw-input" required autocomplete="current-password">
      </div>"""

    consent_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Authorize {escaped_client} — TubeLens OSS</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    :root {{
      --bg: #0c0406;
      --card-bg: rgba(24, 10, 15, 0.88);
      --border: rgba(244, 63, 94, 0.3);
      --crimson: #ff1744;
      --rose: #f43f5e;
      --amber: #fb923c;
      --text: #fdf2f4;
      --muted: #fda4af;
      --muted-sub: #9f1239;
      --error: #fb7185;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    body {{
      background: radial-gradient(circle at 50% 20%, #200810 0%, var(--bg) 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 1.25rem;
      padding: 2.5rem 2rem;
      max-width: 440px;
      width: 100%;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7), 0 0 40px rgba(255, 23, 68, 0.15);
      backdrop-filter: blur(16px);
      text-align: center;
    }}
    .logo {{
      width: 56px;
      height: 56px;
      margin: 0 auto 1.25rem;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255, 23, 68, 0.1);
      border: 1px solid rgba(255, 23, 68, 0.35);
      border-radius: 1rem;
      box-shadow: 0 0 20px rgba(255, 23, 68, 0.2);
    }}
    h1 {{ font-size: 1.35rem; font-weight: 700; margin-bottom: 0.5rem; color: #fff; }}
    p.subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.75rem; line-height: 1.5; }}
    .permissions {{
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(244, 63, 94, 0.15);
      border-radius: 0.75rem;
      padding: 1rem;
      text-align: left;
      margin-bottom: 2rem;
    }}
    .permission-item {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
      font-size: 0.85rem;
      color: #ffe4e6;
      margin-bottom: 0.5rem;
    }}
    .permission-item:last-child {{ margin-bottom: 0; }}
    .check {{ color: var(--crimson); font-weight: bold; }}
    .password-section {{
      margin-bottom: 1.5rem;
      text-align: left;
    }}
    .pw-label {{
      display: block;
      font-size: 0.8rem;
      color: var(--muted);
      margin-bottom: 0.4rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .pw-input {{
      width: 100%;
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      border: 1px solid rgba(244, 63, 94, 0.3);
      background: rgba(0, 0, 0, 0.5);
      color: var(--text);
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .pw-input:focus {{
      border-color: var(--crimson);
      box-shadow: 0 0 0 2px rgba(255, 23, 68, 0.25);
    }}
    .error-msg {{
      background: rgba(251, 113, 133, 0.15);
      border: 1px solid rgba(251, 113, 133, 0.35);
      border-radius: 0.5rem;
      padding: 0.6rem 0.85rem;
      margin-bottom: 0.75rem;
      font-size: 0.85rem;
      color: var(--error);
    }}
    .actions {{ display: flex; gap: 0.75rem; }}
    button, a.btn-cancel {{
      flex: 1;
      padding: 0.85rem 1rem;
      border-radius: 0.65rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s ease;
    }}
    .btn-approve {{
      background: linear-gradient(135deg, var(--crimson), var(--amber));
      color: #ffffff;
      border: none;
      font-weight: 700;
      box-shadow: 0 4px 15px rgba(255, 23, 68, 0.35);
    }}
    .btn-approve:hover {{
      transform: translateY(-1px);
      box-shadow: 0 6px 22px rgba(255, 23, 68, 0.55);
    }}
    .btn-cancel {{
      background: rgba(255, 255, 255, 0.05);
      color: var(--muted);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .btn-cancel:hover {{
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
    }}
    .footer {{
      margin-top: 1.5rem;
      font-size: 0.75rem;
      color: #9f1239;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">
      <svg width="34" height="34" viewBox="0 0 64 64" fill="none">
        <circle cx="32" cy="32" r="22" stroke="#ff1744" stroke-width="3" stroke-dasharray="14 4 4 4"/>
        <circle cx="32" cy="32" r="14" stroke="#f43f5e" stroke-width="2"/>
        <path d="M32 6v6M32 52v6M6 32h6M52 32h6" stroke="#fb7185" stroke-width="2" stroke-linecap="round"/>
        <circle cx="32" cy="32" r="5" fill="#ff1744"/>
        <circle cx="30" cy="30" r="1.5" fill="#ffffff"/>
      </svg>
    </div>
    <h1>Connect {escaped_client}</h1>
    <p class="subtitle"><strong>{escaped_client}</strong> wants to connect to your <strong>TubeLens OSS</strong> server to research and cite YouTube videos.</p>

    <div class="permissions">
      <div class="permission-item"><span class="check">✓</span> YouTube channel resolution &amp; catalogs</div>
      <div class="permission-item"><span class="check">✓</span> Multimodal video summaries &amp; fact checks</div>
      <div class="permission-item"><span class="check">✓</span> Cross-video consensus &amp; Q&amp;A citations</div>
    </div>

    <form method="POST">
      <input type="hidden" name="client_id" value="{client_id}">
      <input type="hidden" name="redirect_uri" value="{escaped_redirect}">
      <input type="hidden" name="state" value="{escaped_state}">
      <input type="hidden" name="scope" value="{escaped_scope}">
      <input type="hidden" name="code_challenge" value="{escaped_challenge}">
      <input type="hidden" name="code_challenge_method" value="{escaped_method}">

      {password_field_html}

      <div class="actions">
        <a href="{escaped_redirect}?error=access_denied&state={escaped_state}" class="btn-cancel">Cancel</a>
        <button type="submit" class="btn-approve">Authorize</button>
      </div>
    </form>

    <div class="footer">
      Zero Media Storage • 100% Stateless • Fair Use Compliant
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(consent_html)


async def handle_oauth_token(request: Request) -> Response:
    """
    RFC 6749 Token Endpoint.
    Supports authorization_code exchange and refresh_token grants.
    """
    # Support both application/x-www-form-urlencoded and application/json
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}
    else:
        form = await request.form()
        body = dict(form)

    # Also parse HTTP Basic Auth for client credentials
    client_id = body.get("client_id")
    client_secret = body.get("client_secret")
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            if ":" in decoded:
                b_user, b_pass = decoded.split(":", 1)
                client_id = client_id or b_user
                client_secret = client_secret or b_pass
        except Exception:
            pass

    grant_type = body.get("grant_type", "")
    issuer = oauth_server.get_issuer_url(request)

    try:
        if grant_type == "authorization_code":
            code = body.get("code", "")
            redirect_uri = body.get("redirect_uri")
            code_verifier = body.get("code_verifier")
            if not code:
                return JSONResponse({"error": "invalid_request", "error_description": "Missing code"}, status_code=400)

            token_data = oauth_server.exchange_code(
                code=code,
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
                issuer=issuer,
            )
            return JSONResponse(token_data)

        elif grant_type == "refresh_token":
            refresh_token = body.get("refresh_token", "")
            if not refresh_token:
                return JSONResponse(
                    {"error": "invalid_request", "error_description": "Missing refresh_token"}, status_code=400
                )

            token_data = oauth_server.refresh_access_token(refresh_token, issuer=issuer)
            return JSONResponse(token_data)

        else:
            return JSONResponse(
                {"error": "unsupported_grant_type", "error_description": f"Unsupported grant_type: '{grant_type}'"},
                status_code=400,
            )

    except ValueError as ve:
        err_msg = str(ve)
        err_code = "invalid_grant"
        if ":" in err_msg:
            err_code, err_msg = [s.strip() for s in err_msg.split(":", 1)]
        return JSONResponse({"error": err_code, "error_description": err_msg}, status_code=400)
    except Exception as e:
        logger.error(f"Token generation error: {e}", exc_info=True)
        return JSONResponse(
            {"error": "server_error", "error_description": "Internal token generation error"}, status_code=500
        )


async def handle_oauth_userinfo(request: Request) -> JSONResponse:
    """
    OpenID Connect UserInfo / Token Introspection Endpoint.
    Returns subject info for authorized Bearer tokens.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    token = auth_header.split(" ", 1)[1]
    payload = oauth_server.verify_token(token)
    if not payload:
        return JSONResponse({"error": "invalid_token"}, status_code=401)

    return JSONResponse(
        {
            "sub": payload.get("sub", "claude-user"),
            "scope": payload.get("scope", "tubelens"),
            "iss": payload.get("iss"),
            "active": True,
        }
    )
