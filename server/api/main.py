import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from core.telemetry.aggregator import MetricsAggregator

from core.queue.request import InferenceRequest
from core.scheduler.batch_scheduler import DynamicBatchScheduler
from inference.service import InferenceService


app = FastAPI(
    title="GPUFlow-X",
    description="AI-Driven Self-Learning GPU Inference Scheduler",
    version="0.1.0"
)


# Create inference service
inference_service = InferenceService()


# Create dynamic batch scheduler
scheduler = DynamicBatchScheduler(
    inference_service=inference_service,
    max_queue_size=100
)


# Store requests for status tracking
request_store = {}

request_store_lock = threading.Lock()


class InferenceAPIRequest(BaseModel):
    values: list[float] = Field(
        ...,
        min_length=10,
        max_length=10
    )


@app.on_event("startup")
def startup():
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    scheduler.stop()


@app.get("/")
def root():
    return {
        "project": "GPUFlow-X",
        "status": "running",
        "phase": "Phase 3 - Dynamic Batching"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "queue_size": scheduler.queue_size()
    }


@app.post("/infer")
def infer(request: InferenceAPIRequest):

    inference_request = InferenceRequest(
        values=request.values
    )

    request_id = scheduler.submit(
        inference_request
    )

    with request_store_lock:
        request_store[request_id] = inference_request

    return {
        "request_id": request_id,
        "status": inference_request.status,
        "queue_size": scheduler.queue_size()
    }


@app.get("/requests/{request_id}")
def get_request_status(request_id: str):

    with request_store_lock:
        inference_request = request_store.get(
            request_id
        )

    if inference_request is None:
        raise HTTPException(
            status_code=404,
            detail="Request not found."
        )

    response = {
        "request_id": inference_request.request_id,
        "status": inference_request.status,
        "queue_time_ms": inference_request.queue_time_ms,
        "inference_time_ms": inference_request.inference_time_ms,
        "total_latency_ms": inference_request.total_latency_ms
    }

    if inference_request.status == "completed":
        response["device"] = "cpu"

    return response


@app.get("/metrics")
def get_metrics():
    events = scheduler.telemetry.get_events()

    aggregator = MetricsAggregator(events)

    return {
        "scheduler": scheduler.metrics(),
        "performance": aggregator.calculate(),
    }
