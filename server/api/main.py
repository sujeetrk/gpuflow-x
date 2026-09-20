import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai.latency_predictor.predictor import LatencyPredictor
from ai.ops_copilot.copilot import AIOpsCopilot
from core.queue.request import InferenceRequest
from core.scheduler.batch_scheduler import DynamicBatchScheduler
from inference.service import InferenceService


app = FastAPI(
    title="GPUFlow-X",
    description="AI-Driven Self-Learning GPU Inference Scheduler",
    version="0.1.0",
)


# ---------------------------------------------------------
# Core Services
# ---------------------------------------------------------

inference_service = InferenceService()

scheduler = DynamicBatchScheduler(
    inference_service=inference_service,
    max_queue_size=100,
)

latency_predictor = LatencyPredictor()
ops_copilot = AIOpsCopilot()

request_store = {}

request_store_lock = threading.Lock()


# ---------------------------------------------------------
# API Request Models
# ---------------------------------------------------------

class InferenceAPIRequest(BaseModel):
    values: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
    )


# ---------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------

@app.on_event("startup")
def startup():
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    scheduler.stop()


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "GPUFlow-X",
        "status": "running",
        "phase": "Phase 7 - AI Latency Prediction",
    }


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "queue_size": scheduler.queue_size(),
    }


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

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
        "queue_size": scheduler.queue_size(),
    }


# ---------------------------------------------------------
# Request Status
# ---------------------------------------------------------

@app.get("/requests/{request_id}")
def get_request_status(request_id: str):

    with request_store_lock:

        inference_request = request_store.get(
            request_id
        )

    if inference_request is None:

        raise HTTPException(
            status_code=404,
            detail="Request not found.",
        )

    response = {
        "request_id": inference_request.request_id,
        "status": inference_request.status,
        "queue_time_ms": inference_request.queue_time_ms,
        "inference_time_ms": inference_request.inference_time_ms,
        "total_latency_ms": inference_request.total_latency_ms,
    }

    if inference_request.status == "completed":

        response["device"] = "cpu"

    return response


# ---------------------------------------------------------
# Scheduler Metrics
# ---------------------------------------------------------

@app.get("/metrics")
def get_metrics():

    return scheduler.metrics()


# ---------------------------------------------------------
# AI Latency Prediction
# ---------------------------------------------------------

@app.get("/predict/latency")
def predict_latency(
    batch_size: int,
    batch_delay_ms: float,
    request_count: int,
):

    if batch_size < 1:

        raise HTTPException(
            status_code=400,
            detail="batch_size must be at least 1.",
        )

    if batch_delay_ms < 0:

        raise HTTPException(
            status_code=400,
            detail="batch_delay_ms cannot be negative.",
        )

    if request_count < 1:

        raise HTTPException(
            status_code=400,
            detail="request_count must be at least 1.",
        )

    predicted_latency_ms = latency_predictor.predict(
        batch_size=batch_size,
        batch_delay_ms=batch_delay_ms,
        request_count=request_count,
    )

    return {
        "batch_size": batch_size,
        "batch_delay_ms": batch_delay_ms,
        "request_count": request_count,
        "predicted_latency_ms": predicted_latency_ms,
        "model": type(
            latency_predictor.model
        ).__name__,
    }


# ---------------------------------------------------------
# AI Operations Copilot
# ---------------------------------------------------------

@app.get("/ops/copilot")
def get_ops_copilot_report():
    """Explain current scheduler health and observed bottlenecks."""
    metrics = scheduler.metrics()
    events = scheduler.telemetry.get_events()

    return ops_copilot.analyze(
        metrics=metrics,
        telemetry_events=events,
    )


@app.get("/ops/copilot/decision")
def get_ops_copilot_decision():
    """Explain the latest scheduler decision."""
    metrics = scheduler.metrics()

    return {
        "decision": metrics.get("last_ai_decision"),
        "explanation": ops_copilot.explain_decision(
            metrics.get("last_ai_decision")
        ),
    }
