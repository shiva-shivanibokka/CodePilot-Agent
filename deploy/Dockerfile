# The service.
#
# Read DEPLOYING.md before using this: a container that spawns sandbox
# containers needs the Docker socket, and handing a container the Docker socket
# is equivalent to giving it root on the host. Running the service directly on
# a host that has Docker avoids that entirely, and is the recommended shape.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt "uvicorn[standard]>=0.30"

COPY codepilot ./codepilot
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Refuses to start without CODEPILOT_SERVER_TOKEN and CODEPILOT_WORKSPACE_ROOT.
ENV CODEPILOT_WORKSPACE_ROOT=/repos
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["python", "-m", "codepilot.server", "--host", "0.0.0.0", "--port", "8000"]
