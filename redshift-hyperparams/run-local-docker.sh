#!/bin/bash

./builddocker.sh

export IMAGE_REPO_NAME=redshift_nn
export IMAGE_TAG=0.1

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
IMAGE_URI="gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG"
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# make sure that the job directory is created before mounting to Docker
mkdir -p ./tmp/job/logs

docker run -it --rm \
    -v "${CWD}/data/":/root/data/ \
    -v "${CWD}/tmp/job/":/root/job \
    "$IMAGE_URI" "$@"
