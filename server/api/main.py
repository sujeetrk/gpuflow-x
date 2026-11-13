import logging
import math
import threading
import time

from collections import OrderedDict, deque
from queue import Full
from typing import Optional

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ai.latency_predictor.predictor import LatencyPredictor
from ai.ops_copilot.copilot import AIOpsCopilot
from core.queue.request import InferenceRequest
from core.scheduler.ai_scheduler import AIScheduler
from inference.service import InferenceService
from inference.gpu_monitor import GPUHealthMonitor


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------

RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_CLIENTS = 10000


class SlidingWindowRateLimiter:
    """Thread-safe, bounded, in-memory sliding-window limiter."""

    def __init__(
        self,
        limit=RATE_LIMIT_REQUESTS,
        window_seconds=RATE_LIMIT_WINDOW_SECONDS,
        max_clients=RATE_LIMIT_MAX_CLIENTS,
    ):
        if limit < 1 or window_seconds <= 0 or max_clients < 1:
            raise ValueError("Invalid rate limiter configuration.")

        self.limit = limit
        self.window_seconds = float(window_seconds)
        self.max_clients = max_clients
        self._clients = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, client_id):
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            # Remove inactive clients from the oldest end.
            while self._clients:
                oldest_id, oldest_requests = next(
                    iter(self._clients.items())
                )

                while (
                    oldest_requests
                    and oldest_requests[0] <= cutoff
                ):
                    oldest_requests.popleft()

                if oldest_requests:
                    break

                self._clients.pop(oldest_id, None)

            requests = self._clients.get(client_id)

            if requests is None:
                if len(self._clients) >= self.max_clients:
                    self._clients.popitem(last=False)

                requests = deque()
                self._clients[client_id] = requests

            while requests and requests[0] <= cutoff:
                requests.popleft()

            if len(requests) >= self.limit:
                retry_after = max(
                    1,
                    math.ceil(
                        self.window_seconds - (now - requests[0])
                    ),
                )
                self._clients.move_to_end(client_id)
                return False, retry_after

            requests.append(now)
            self._clients.move_to_end(client_id)

            return True, 0


# ---------------------------------------------------------
# Core Services
# ---------------------------------------------------------

inference_service = InferenceService()

scheduler = AIScheduler(
    inference_service=inference_service,
    max_queue_size=100,
    learning_enabled=True,
    telemetry_storage_path="data/gpuflow_telemetry.db",
)

latency_predictor = LatencyPredictor()
ops_copilot = AIOpsCopilot()


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="GPUFlow-X",
    description="AI-Driven Self-Learning GPU Inference Scheduler",
    version="0.1.0",
)

app.state.rate_limiter = SlidingWindowRateLimiter()


# ---------------------------------------------------------
# Security Middleware
# ---------------------------------------------------------

@app.middleware("http")
async def security_middleware(request, call_next):
    # Rate-limit inference submissions by the connecting client's IP.
    if request.method == "POST" and request.url.path == "/infer":
        client_id = (
            request.client.host
            if request.client is not None
            else "unknown"
        )

        allowed, retry_after = app.state.rate_limiter.allow(
            client_id
        )

        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Retry later."
                },
                headers={
                    "Retry-After": str(retry_after),
                },
            )
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )

    return response


# ---------------------------------------------------------
# API Request Models
# ---------------------------------------------------------

class InferenceAPIRequest(BaseModel):
    values: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
    )

    sla_budget_ms: Optional[float] = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    @field_validator("values")
    @classmethod
    def validate_finite_values(cls, values):
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "All inference values must be finite numbers."
            )
        return values


# ---------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------

@app.on_event("startup")
def startup():
    scheduler.start()
    logger.info("GPUFlow-X API started")


@app.on_event("shutdown")
def shutdown():
    scheduler.stop()
    logger.info("GPUFlow-X API stopped")


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "GPUFlow-X",
        "status": "running",
        "phase": "Production AI Scheduling",
    }


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "queue_size": scheduler.queue.size(),
    }


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

@app.post("/infer")
def infer(request: InferenceAPIRequest):
    inference_request = InferenceRequest(
        values=request.values,
        sla_budget_ms=request.sla_budget_ms,
    )

    try:
        request_id = scheduler.submit(inference_request)

    except Full:
        raise HTTPException(
            status_code=429,
            detail="Inference queue is full. Retry later.",
            headers={"Retry-After": "1"},
        )

    return {
        "request_id": request_id,
        "status": inference_request.status,
        "queue_size": scheduler.queue.size(),
    }


# ---------------------------------------------------------
# Request Status
# ---------------------------------------------------------

@app.get("/requests/{request_id}")
def get_request_status(
    request_id: str = Path(
        min_length=1,
        max_length=64,
    ),
):
    result = scheduler.get_request(request_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Request not found.",
        )

    return result


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
    if batch_size < 1 or batch_size > 1024:
        raise HTTPException(
            status_code=400,
            detail="batch_size must be between 1 and 1024.",
        )

    if (
        not math.isfinite(batch_delay_ms)
        or batch_delay_ms < 0
        or batch_delay_ms > 60000
    ):
        raise HTTPException(
            status_code=400,
            detail="batch_delay_ms must be finite and between 0 and 60000.",
        )

    if request_count < 1 or request_count > 100000:
        raise HTTPException(
            status_code=400,
            detail="request_count must be between 1 and 100000.",
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
        "model": type(latency_predictor.model).__name__,
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

# ---------------------------------------------------------
# GPU Health Monitoring
# ---------------------------------------------------------

@app.get("/gpu/health")
def get_gpu_health():
    """Return GPU availability, device assignments, and memory health."""
    monitor = GPUHealthMonitor(
        scheduler.inference_service.device_manager
    )
    return monitor.snapshot()
