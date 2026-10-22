#!/usr/bin/env bash
# Deploy simfull-app to Google Cloud Run.
# Run in Google Cloud Shell (shell.cloud.google.com), from this repository's root:
#   GITHUB_TOKEN=<token> bash deploy/cloud-run.sh
#
# Environment variables:
#   GITHUB_TOKEN: GitHub personal access token with read access to Frazcoding/simcore (private)
#   REGION: Google Cloud region (default: europe-west1)
#   SERVICE: Cloud Run service name (default: simfull-app)
#   SIMCORE_REF: branch or tag of simcore to clone (default: main)
#   PROJECT: Google Cloud project ID (default: gcloud's currently configured project)
#   SKIP_SETUP: set to 1 to skip enabling APIs & IAM role granting (used by GitHub Actions)
#   IMAGE: deploy an existing pre-built container image instead of building from source
set -euo pipefail
cd "$(dirname "$0")/.."
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-simfull-app}"
SIMCORE_REF="${SIMCORE_REF:-main}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

if [ -z "$PROJECT" ]; then
  echo "No Google Cloud project selected: run  gcloud config set project <PROJECT_ID>  first." >&2
  exit 1
fi

if [ -z "${IMAGE:-}" ]; then
  echo "== simcore ($SIMCORE_REF) into vendor/simcore"
  rm -rf vendor/simcore
  url="https://github.com/Frazcoding/simcore"
  [ -n "${GITHUB_TOKEN:-}" ] && url="https://x-access-token:${GITHUB_TOKEN}@github.com/Frazcoding/simcore"
  git clone --quiet --depth 1 --branch "$SIMCORE_REF" "$url" vendor/simcore
  git -C vendor/simcore log --oneline -1
fi

if [ -z "${SKIP_SETUP:-}" ]; then
  echo "== APIs (first time only)"
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
    --project "$PROJECT" --quiet

  number="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')"
  builder="serviceAccount:${number}-compute@developer.gserviceaccount.com"
  if ! gcloud projects get-iam-policy "$PROJECT" --flatten bindings[].members \
       --filter "bindings.role=roles/run.builder AND bindings.members=$builder" --format 'value(bindings.role)' | grep -q .; then
    if gcloud projects add-iam-policy-binding "$PROJECT" --quiet --condition=None --member "$builder" \
         --role roles/run.builder >/dev/null; then
      echo "Granted the build account the Cloud Run Builder role; waiting 2 minutes for it to apply..."
      sleep 120
    else
      echo "(could not grant roles/run.builder to $builder; if the build is refused, grant it in IAM)"
    fi
  fi
fi

echo "== build and deploy $SERVICE to $PROJECT / $REGION"
if [ -n "${IMAGE:-}" ]; then from=(--image "$IMAGE"); else from=(--source .); fi

gcloud run deploy "$SERVICE" "${from[@]}" --project "$PROJECT" --region "$REGION" --quiet \
  --allow-unauthenticated --memory 1Gi --cpu 1 --min-instances 0 --max-instances 2 --concurrency 8 \
  --timeout 120 --set-env-vars SIMFULL_APP_HOST=0.0.0.0

echo
echo "simfull-app: $(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
