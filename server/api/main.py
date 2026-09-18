from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference.service import InferenceService


app = FastAPI(
    title="GPUFlow-X",
    description="AI-Driven Self-Learning GPU Inference Scheduler",
    version="0.1.0"
)

inference_service = InferenceService()


class InferenceRequest(BaseModel):
    values: list[float] = Field(..., min_length=10, max_length=10)


class BatchInferenceRequest(BaseModel):
    requests: list[InferenceRequest] = Field(
        ...,
        min_length=1,
        max_length=8
    )


@app.get("/")
def root():
    return {
        "project": "GPUFlow-X",
        "status": "running",
        "phase": "Phase 1 - GPU Inference Engine"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/infer")
def infer(request: InferenceRequest):
    return inference_service.infer(request.values)


@app.post("/infer/batch")
def infer_batch(request: BatchInferenceRequest):
    batch_values = [
        item.values
        for item in request.requests
    ]

    return inference_service.infer_batch(batch_values)
