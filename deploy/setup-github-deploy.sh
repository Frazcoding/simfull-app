#!/usr/bin/env bash
# One-time set-up so GitHub Actions (.github/workflows/deploy.yml) can deploy to Cloud Run with no key file:
# a deploy service account that only workflows of Frazcoding/simfull-app may act as (Workload Identity Federation).
# Run once in Google Cloud Shell:   bash deploy/setup-github-deploy.sh
# Safe to run again: existing pieces are kept.
set -euo pipefail
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REPO="${REPO:-Frazcoding/simfull-app}"
SA_NAME=github-deploy
POOL=github
PROVIDER=simfull-app

if [ -z "$PROJECT" ]; then
  echo "Run  gcloud config set project <PROJECT_ID>  first." >&2
  exit 1
fi

retry() {
  local n
  for n in 1 2 3 4 5; do
    "$@" && return 0
    echo "  (busy, retrying in $((n * 15)) s)" >&2
    sleep $((n * 15))
  done
  "$@"
}

NUMBER="$(retry gcloud projects describe "$PROJECT" --format 'value(projectNumber)')"
SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

echo "== APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com --project "$PROJECT" --quiet

echo "== deploy service account $SA"
gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" --display-name "GitHub deploy (simfull-app)"

for role in roles/run.admin roles/cloudbuild.builds.editor roles/storage.admin roles/artifactregistry.writer \
            roles/iam.serviceAccountUser roles/serviceusage.serviceUsageConsumer roles/logging.viewer; do
  retry gcloud projects add-iam-policy-binding "$PROJECT" --quiet --condition=None \
    --member "serviceAccount:$SA" --role "$role" >/dev/null
  sleep 3
done

retry gcloud projects add-iam-policy-binding "$PROJECT" --quiet --condition=None \
  --member "serviceAccount:${NUMBER}-compute@developer.gserviceaccount.com" --role roles/run.builder >/dev/null

echo "== GitHub identity pool (for $REPO)"
gcloud iam workload-identity-pools describe "$POOL" --location global --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam workload-identity-pools create "$POOL" --location global --project "$PROJECT" --display-name "GitHub"

gcloud iam workload-identity-pools providers describe "$PROVIDER" --workload-identity-pool "$POOL" --location global \
    --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --workload-identity-pool "$POOL" \
       --location global --project "$PROJECT" --display-name "simfull-app" \
       --issuer-uri "https://token.actions.githubusercontent.com" \
       --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
       --attribute-condition "assertion.repository == '$REPO'"

retry gcloud iam service-accounts add-iam-policy-binding "$SA" --project "$PROJECT" --quiet \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${REPO}" \
  >/dev/null

echo
echo "Setup complete for $REPO on $PROJECT."
