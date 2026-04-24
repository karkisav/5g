# Docker Guide

This project runs the Flask server (`server.py`) on port `5050`.

Docker files are organized under the `docker/` folder:

- `docker/Dockerfile`
- `docker/compose.yml`
- `docker/compose.camera-linux.yml`

## 1) Build the image

```bash
docker build -f docker/Dockerfile -t faceos:latest .
```

## 2) Run with plain Docker

```bash
mkdir -p data
docker run --rm -it \
  -p 5050:5050 \
  -e INSIGHTFACE_PROVIDERS=CPUExecutionProvider \
  -e EMBEDDINGS_FILE=/app/data/embeddings.npy \
  -e NAMES_FILE=/app/data/names.json \
  -v "$PWD/data:/app/data" \
  --name faceos \
  faceos:latest
```

Open: `http://localhost:5050`

## 3) Run with Docker Compose (recommended)

```bash
mkdir -p data
docker compose -f docker/compose.yml up --build
```

Detached mode:

```bash
docker compose -f docker/compose.yml up --build -d
```

Stop:

```bash
docker compose -f docker/compose.yml down
```

## 4) Linux webcam passthrough (optional)

If your host webcam is at `/dev/video0`:

```bash
docker compose -f docker/compose.yml -f docker/compose.camera-linux.yml up --build
```

## 5) Notes for macOS

Docker Desktop on macOS typically does not pass through the host webcam directly to Linux containers. The containerized server can still run, but live camera capture may not work inside the container on macOS.
