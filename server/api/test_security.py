import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import server.api.main as api_module


VALID_VALUES = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0,
]


class FakeScheduler:
    def __init__(self):
        self.submitted_request = None

    def start(self):
        pass

    def stop(self):
        pass

    def submit(self, request):
        self.submitted_request = request
        return request.request_id

    @property
    def queue(self):
        class Queue:
            @staticmethod
            def size():
                return 0

        return Queue()


@pytest.fixture
def security_client(monkeypatch):
    fake_scheduler = FakeScheduler()

    monkeypatch.setattr(
        api_module,
        "scheduler",
        fake_scheduler,
    )

    api_module.app.state.rate_limiter = (
        api_module.SlidingWindowRateLimiter(
            limit=60,
            window_seconds=60,
        )
    )

    with TestClient(api_module.app) as client:
        yield client, fake_scheduler


def test_security_headers_are_present(security_client):
    client, _ = security_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "camera=()" in response.headers["permissions-policy"]


def test_inference_endpoint_still_accepts_valid_request(
    security_client,
):
    client, scheduler = security_client

    response = client.post(
        "/infer",
        json={"values": VALID_VALUES},
    )

    assert response.status_code == 200
    assert scheduler.submitted_request is not None


@pytest.mark.parametrize(
    "values",
    [
        [float("nan")] * 10,
        [float("inf")] * 10,
        [float("-inf")] * 10,
    ],
)
def test_inference_model_rejects_non_finite_values(values):
    with pytest.raises(ValidationError):
        api_module.InferenceAPIRequest.model_validate(
            {"values": values}
        )


def test_rate_limiter_blocks_excess_requests():
    limiter = api_module.SlidingWindowRateLimiter(
        limit=2,
        window_seconds=60,
    )

    assert limiter.allow("client-a")[0] is True
    assert limiter.allow("client-a")[0] is True

    allowed, retry_after = limiter.allow("client-a")

    assert allowed is False
    assert retry_after >= 1


def test_rate_limiter_tracks_clients_independently():
    limiter = api_module.SlidingWindowRateLimiter(
        limit=1,
        window_seconds=60,
    )

    assert limiter.allow("client-a")[0] is True
    assert limiter.allow("client-a")[0] is False
    assert limiter.allow("client-b")[0] is True


def test_rate_limiter_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        api_module.SlidingWindowRateLimiter(limit=0)

    with pytest.raises(ValueError):
        api_module.SlidingWindowRateLimiter(window_seconds=0)


def test_rate_limiter_bounds_client_storage():
    limiter = api_module.SlidingWindowRateLimiter(
        limit=5,
        window_seconds=60,
        max_clients=2,
    )

    limiter.allow("client-a")
    limiter.allow("client-b")
    limiter.allow("client-c")

    assert len(limiter._clients) <= 2


@pytest.mark.parametrize(
    "batch_size,batch_delay_ms,request_count",
    [
        (0, 1.0, 1),
        (1025, 1.0, 1),
        (1, -1.0, 1),
        (1, float("inf"), 1),
        (1, float("nan"), 1),
        (1, 60001.0, 1),
        (1, 1.0, 0),
        (1, 1.0, 100001),
    ],
)
def test_latency_prediction_rejects_invalid_parameters(
    security_client,
    batch_size,
    batch_delay_ms,
    request_count,
):
    client, _ = security_client

    response = client.get(
        "/predict/latency",
        params={
            "batch_size": batch_size,
            "batch_delay_ms": batch_delay_ms,
            "request_count": request_count,
        },
    )

    assert response.status_code in (400, 422)
