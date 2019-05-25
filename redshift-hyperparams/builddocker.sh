#!/bin/bash

export IMAGE_REPO_NAME=redshift_nn
export IMAGE_TAG=0.1

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_URI="gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG"

export PROJECT_ID
export IMAGE_URI

docker build -f Dockerfile -t "$IMAGE_URI" ./
