#!/bin/bash

export BUCKET_NAME=wgauvin-learn-cloudml
export REGION=us-east1
export IMAGE_REPO_NAME=redshift_nn
export IMAGE_TAG=0.1

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_URI="gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG"
JOB_NAME="redshift_nn_train_$(date +%Y%m%d_%H%M%S)"
JOB_DIR="gs://$BUCKET_NAME/$JOB_NAME"
TRAIN_FILE="gs://$BUCKET_NAME/data/astromonical_data.csv.gz"

export PROJECT_ID
export IMAGE_URI
export JOB_NAME
export JOB_DIR

# Make sure latest docker has been pushed
docker push "$IMAGE_URI"

gcloud beta ml-engine jobs submit training "$JOB_NAME" \
  --scale-tier BASIC \
  --region "$REGION" \
  --master-image-uri "$IMAGE_URI" \
  --config training-config.yml \
  -- \
  --job-dir "${JOB_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --hypertuning 1 \
  --epochs 10
