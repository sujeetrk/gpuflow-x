# GPUFlow-X Deployment

GPUFlow-X supports CPU deployment and optional NVIDIA CUDA deployment.

## Requirements

- Docker Engine or Docker Desktop with Docker Compose.
- For GPU deployment: NVIDIA GPU, NVIDIA drivers, and NVIDIA Container Toolkit.

## CPU deployment

Build and start:

    docker compose up --build -d api

Check health:

    curl http://localhost:8000/health

Check metrics:

    curl http://localhost:8000/metrics

Submit inference:

    curl -X POST http://localhost:8000/infer \
      -H "Content-Type: application/json" \
      -d '{"values":[0,1,2,3,4,5,6,7,8,9]}'

View logs:

    docker compose logs -f api

Stop:

    docker compose down

The SQLite telemetry database is stored in the persistent
gpuflow-data Docker volume mounted at /app/data.

## NVIDIA CUDA deployment

Build and start the CUDA service:

    docker compose --profile cuda up --build -d api-cuda

Check health:

    curl http://localhost:8000/health

Verify CUDA inside the container:

    docker compose exec api-cuda python3 -c \
      "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

CUDA execution requires compatible NVIDIA hardware and the NVIDIA
Container Toolkit.

Stop:

    docker compose --profile cuda down

## Direct Docker builds

CPU:

    docker build -f deployment/Dockerfile -t gpuflow-x:cpu .

CUDA:

    docker build -f deployment/Dockerfile.cuda -t gpuflow-x:cuda .

## Health monitoring

Both images define Docker health checks against:

    http://127.0.0.1:8000/health

The API also exposes:

- /health
- /gpu/health
- /metrics

## Continuous integration

GitHub Actions runs the Python test suite and builds the CPU Docker image
on pushes and pull requests targeting main.

CUDA image execution requires a suitable NVIDIA-enabled runner and is
not part of the standard CPU CI job.

## Deployment files

- Dockerfile: CPU image.
- Dockerfile.cuda: NVIDIA CUDA image.
- requirements-cuda.txt: CUDA dependencies.
- docker-compose.yml: CPU and CUDA deployment services.
- .dockerignore: root Docker build exclusions.
- .github/workflows/ci.yml: automated CI pipeline.

## Validation status

YAML configuration and deployment-file checks have been performed
in the development environment.

Docker image builds and container health checks must be executed
on a machine with Docker installed.

Physical NVIDIA GPU execution must be verified on an NVIDIA-enabled
machine. CPU-only environments cannot establish GPU execution.
