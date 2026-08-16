#!/usr/bin/env bash
# ==============================================================================
# TubeLens OSS — Google Cloud Run Automated Deployment Script
# Project ID: tubelens-oss
# Region: us-central1
# ==============================================================================

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-tubelens-oss}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-tubelens-oss}"
REPO_NAME="mcp-servers"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/tubelens:latest"
OAUTH_ADMIN_PASSWORD="${OAUTH_ADMIN_PASSWORD:-}"

echo "🚀 Starting TubeLens OSS Deployment to Google Cloud Run..."
echo "• Project ID: ${PROJECT_ID}"
echo "• Region: ${REGION}"
echo "• Service Name: ${SERVICE_NAME}"
echo "• Image: ${IMAGE_NAME}"
echo "• Admin Password: $([ -n "${OAUTH_ADMIN_PASSWORD}" ] && echo '[CONFIGURED]' || echo '[NOT CONFIGURED - OPEN ACCESS]')"

# 1. Ensure required Google Cloud APIs are enabled
echo "📦 Verifying enabled Google Cloud APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  generativelanguage.googleapis.com \
  youtube.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Ensure Artifact Registry repository exists
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "📦 Creating Artifact Registry repository '${REPO_NAME}' in ${REGION}..."
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="TubeLens MCP servers docker repository"
fi

# 3. Build container with Cloud Build
echo "🔨 Building and pushing container image via Cloud Build..."
gcloud builds submit . \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_NAME}"

# 4. Deploy to Cloud Run
echo "⚡ Deploying service to Google Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --concurrency=80 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=10 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,GEMINI_PRIMARY_MODEL=gemini-3.7-flash,GEMINI_FALLBACK_MODEL=gemini-3.6-flash,AUTH_MODE=none,OAUTH_ADMIN_PASSWORD=${OAUTH_ADMIN_PASSWORD}"

# 5. Retrieve deployed service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')

echo ""
echo "=============================================================================="
echo "✅ TubeLens OSS Deployed Successfully!"
echo "• Service URL: ${SERVICE_URL}"
echo "• Documentation: ${SERVICE_URL}/docs"
echo "• FastMCP Endpoint: ${SERVICE_URL}/mcp"
echo "• SSE Stream: ${SERVICE_URL}/sse"
echo "• Health Check: ${SERVICE_URL}/healthz"
echo "• LLM Discovery: ${SERVICE_URL}/llms.txt"
echo "=============================================================================="
