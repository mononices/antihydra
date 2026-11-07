#!/usr/bin/env bash
# Build the antihydra sandbox image and ensure the docker network exists.
#
# Usage: bash docker/build_sandbox.sh [image-tag]
#
# Extra docker build args can be appended via EXTRA_BUILD_ARGS, e.g.:
#   EXTRA_BUILD_ARGS="--build-arg WITH_HEAVY=0" bash docker/build_sandbox.sh
set -euo pipefail

IMAGE="${1:-ctfenv:antihydra}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t "$IMAGE" ${EXTRA_BUILD_ARGS:-} "$HERE/sandbox"

if ! docker network inspect ctfnet >/dev/null 2>&1; then
    docker network create ctfnet >/dev/null
fi

echo "sandbox image ready: $IMAGE"
