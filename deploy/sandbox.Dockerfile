# The image the agent's commands run in.
#
# Deliberately small: an interpreter, a test runner, and nothing that talks to
# the network. Containers built from it are started with --network none, a
# memory cap and a CPU cap, and are removed when the run ends.
#
#   docker build -f deploy/sandbox.Dockerfile -t codepilot-sandbox:latest .
FROM python:3.12-slim

# `timeout` enforces the command deadline inside the container. Wrapping the
# call on the host cancels the wait, not the process.
RUN apt-get update \
    && apt-get install -y --no-install-recommends coreutils git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytest==8.*

WORKDIR /workspace
