FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS docbuilder
WORKDIR /docs

COPY mkdocs.yml .
COPY docs ./docs
COPY src ./src

RUN uv pip install --no-cache-dir -r ./docs/requirements.txt --system && \
    zensical build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
    gcc \
    libc-dev \
    linux-headers-amd64 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /src
COPY src/ .
COPY --from=docbuilder /docs/site /src/docs

RUN uv pip install --system --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache-dir -r requirements.txt && \
    rm requirements.txt
    
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2001","--log-config", "/src/logging_config.yaml"]