import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import server.api.main as api_module


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
def api_client(monkeypatch):
    fake_scheduler = FakeScheduler()

    monkeypatch.setattr(
        api_module,
        "scheduler",
        fake_scheduler,
    )

    with TestClient(api_module.app) as client:
        yield client, fake_scheduler


VALID_VALUES = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0,
]


def test_infer_accepts_sla_budget(api_client):
    client, scheduler = api_client

    response = client.post(
        "/infer",
        json={
            "values": VALID_VALUES,
            "sla_budget_ms": 200,
        },
    )

    assert response.status_code == 200

    request = scheduler.submitted_request

    assert request is not None
    assert request.sla_budget_ms == 200
    assert request.deadline_time is not None

    assert math.isclose(
        request.deadline_time - request.arrival_time,
        0.2,
        abs_tol=0.001,
    )


def test_infer_without_sla_budget_remains_supported(api_client):
    client, scheduler = api_client

    response = client.post(
        "/infer",
        json={"values": VALID_VALUES},
    )

    assert response.status_code == 200

    request = scheduler.submitted_request

    assert request is not None
    assert request.sla_budget_ms is None
    assert request.deadline_time is None


@pytest.mark.parametrize(
    "sla_budget",
    [0, -1, -100],
)
def test_infer_rejects_non_positive_sla_budget(
    api_client,
    sla_budget,
):
    client, scheduler = api_client

    response = client.post(
        "/infer",
        json={
            "values": VALID_VALUES,
            "sla_budget_ms": sla_budget,
        },
    )

    assert response.status_code == 422
    assert scheduler.submitted_request is None


@pytest.mark.parametrize(
    "sla_budget",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_request_model_rejects_non_finite_sla_budget(
    sla_budget,
):
    with pytest.raises(ValidationError):
        api_module.InferenceAPIRequest.model_validate(
            {
                "values": VALID_VALUES,
                "sla_budget_ms": sla_budget,
            }
        )
