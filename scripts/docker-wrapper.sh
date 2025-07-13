#!/bin/bash
# Simple Docker wrapper that ensures DOCKER_HOST is unset for local builds
unset DOCKER_HOST
exec docker "$@"
