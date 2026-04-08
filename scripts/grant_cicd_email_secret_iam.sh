#!/usr/bin/env bash
# Grant Secret Manager access for Cloud Build notify-email (cicd-gmail-app-password).
# Run from anywhere: bash scripts/grant_cicd_email_secret_iam.sh
set -euo pipefail

PROJECT="${GCP_PROJECT:-mlops-489703}"
SECRET="${SMTP_SECRET_NAME:-cicd-gmail-app-password}"

PN="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
echo "Project: ${PROJECT} (number ${PN})"
echo "Secret:  ${SECRET}"
echo ""

grant() {
  local member="$1"
  echo "Granting secretAccessor to ${member} ..."
  gcloud secrets add-iam-policy-binding "${SECRET}" \
    --project="${PROJECT}" \
    --member="${member}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet
}

# Default Cloud Build service account (most GitHub triggers)
grant "serviceAccount:${PN}@cloudbuild.gserviceaccount.com"

# Default Compute SA (some older / mixed setups resolve secrets with this identity)
grant "serviceAccount:${PN}-compute@developer.gserviceaccount.com"

# User-managed SA used by this project’s Cloud Build trigger (org policy / console setting)
grant "serviceAccount:mlops-pipeline@${PROJECT}.iam.gserviceaccount.com"

echo ""
echo "IAM on secret (verify both lines appear):"
gcloud secrets get-iam-policy "${SECRET}" --project="${PROJECT}" --format=yaml

echo ""
echo "If notify-email still fails with PermissionDenied:"
echo "  1) Cloud Console → Cloud Build → Triggers → your trigger → check 'Service account'."
echo "  2) Grant the same role to THAT email on secret ${SECRET}:"
echo "     gcloud secrets add-iam-policy-binding ${SECRET} --project=${PROJECT} \\"
echo "       --member='serviceAccount:YOUR_TRIGGER_SA@...' --role=roles/secretmanager.secretAccessor"
