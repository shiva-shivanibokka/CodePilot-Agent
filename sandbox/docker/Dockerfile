# CodePilot sandbox image — runs untrusted LLM-generated code in isolation
FROM python:3.11-slim

# Security: non-root user
RUN useradd -m -u 1000 sandboxuser

# Install common packages LLM-generated code might need
RUN pip install --no-cache-dir \
    pytest \
    requests \
    numpy \
    pandas \
    pydantic \
    httpx \
    fastapi \
    uvicorn

# /workspace is the only writable directory (mounted as tmpfs in docker run)
WORKDIR /workspace

# Drop to non-root
USER sandboxuser

CMD ["tail", "-f", "/dev/null"]
