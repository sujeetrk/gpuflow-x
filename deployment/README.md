# GPUFlow-X Deployment

GPUFlow-X provides separate CPU and NVIDIA CUDA container configurations.

## CPU Deployment

Build the CPU image:

    docker build -f deployment/Dockerfile -t gpuflow-x:phase13 .

Run the API:

    docker run -d --name gpuflow-x-api -p 8000:8000 gpuflow-x:phase13

Health check:

    curl http://127.0.0.1:8000/health

Metrics:

    curl http://127.0.0.1:8000/metrics

Inference:

    curl -X POST http://127.0.0.1:8000/infer \
      -H "Content-Type: application/json" \
      -d '{"values":[0,1,2,3,4,5,6,7,8,9]}'

Stop the container:

    docker rm -f gpuflow-x-api

## NVIDIA CUDA Deployment

The CUDA deployment uses:

- NVIDIA CUDA 12.6 runtime
- PyTorch 2.14.0+cu126
- FastAPI
- Uvicorn
- GPUFlow-X scheduler and inference stack

Build the CUDA image:

    docker build -f deployment/Dockerfile.cuda -t gpuflow-x:cuda .

On an NVIDIA GPU machine with the NVIDIA Container Toolkit installed:

    docker run -d \
      --gpus all \
      --name gpuflow-x-api \
      -p 8000:8000 \
      gpuflow-x:cuda

Verify the container:

    docker logs gpuflow-x-api

Check the API:

    curl http://127.0.0.1:8000/health

Check inference:

    curl -X POST http://127.0.0.1:8000/infer \
      -H "Content-Type: application/json" \
      -d '{"values":[0,1,2,3,4,5,6,7,8,9]}'

## GPU Verification

Inside an NVIDIA-enabled container:

    docker exec gpuflow-x-api python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

Expected GPU deployment behavior:

    torch: 2.14.0+cu126
    CUDA: 12.6
    CUDA available: True
    CUDA device count: >= 1

## Validation Performed on Development Machine

The following were validated locally:

- CPU Docker image build
- CPU container startup
- FastAPI health endpoint
- Metrics endpoint
- Inference endpoint
- Scheduler execution inside container
- Telemetry generation inside container
- CUDA Docker image build
- CUDA-enabled PyTorch installation
- CUDA 12.6 runtime presence

The development laptop does not contain an NVIDIA GPU. Therefore, physical NVIDIA GPU execution and `torch.cuda.is_available() == True` were not claimed as locally validated.

## Deployment Files

- Dockerfile — CPU deployment
- Dockerfile.cuda — NVIDIA CUDA deployment
- requirements-cuda.txt — CUDA-specific Python dependencies
- .dockerignore — Docker build exclusions
