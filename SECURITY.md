# Security Policy & Operational Guidelines

## 1. Supported Versions

| Version | Supported |
| :--- | :--- |
| `1.0.x` | :white_check_mark: |
| `< 1.0.0`| :x: |

## 2. Reporting a Vulnerability

If you discover a security vulnerability within the TubeLens server:
1. Please do **NOT** open a public GitHub issue.
2. Email the maintainer directly at `security@agentspace.ai` with a detailed description and steps to reproduce.
3. We acknowledge receipt within 24 hours and provide a remediation timeline.

## 3. Credential & Secrets Management

- **Zero Hardcoded Secrets:** No API keys, tokens, or service account credentials may be committed to version control.
- **Google Cloud Secret Manager:** In production Cloud Run environments, `YOUTUBE_API_KEY` and `MCP_API_KEY` are mounted via Secret Manager references:
  ```bash
  --set-secrets YOUTUBE_API_KEY=youtube-api-key:latest
  ```
- **Application Default Credentials (ADC):** Vertex AI, Cloud Storage, Cloud Firestore, and Cloud Tasks utilize IAM role bindings attached to the dedicated runtime service account.

## 4. Authentication Modes

### Mode 1: IAM Enforcement (Default & Recommended)
- Cloud Run service is deployed with `--no-allow-unauthenticated`.
- Local developers and IDE clients connect through `gcloud run services proxy`.
- AI agents authenticate using Google Cloud Identity-Aware Proxy (IAP) or direct OIDC identity tokens.

### Mode 2: API Key Mode (Public / Web Agents)
- Cloud Run service is deployed with `--allow-unauthenticated`.
- The ASGI middleware validates incoming requests against the `X-MCP-API-Key` or `Authorization: Bearer <KEY>` header.
- Token validation uses cryptographic constant-time comparison (`secrets.compare_digest`) to prevent timing side-channel attacks.

## 5. Network & Execution Isolation

- Container runs as an unprivileged non-root user (`appuser`, UID 1000).
- Strict per-instance concurrency semaphores combined with Firestore project-level rate limiters prevent API exhaustion and DoS.
- Storage buckets enforce uniform bucket-level access and server-side encryption.
