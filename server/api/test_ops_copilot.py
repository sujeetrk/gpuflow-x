from fastapi.testclient import TestClient

from server.api.main import app


client = TestClient(app)


def test_ops_copilot_report_endpoint():
    response = client.get("/ops/copilot")

    assert response.status_code == 200
    data = response.json()

    assert "summary" in data
    assert "decision_explanation" in data
    assert "bottlenecks" in data
    assert "recommendations" in data
    assert "observed_metrics" in data


def test_ops_copilot_decision_endpoint():
    response = client.get("/ops/copilot/decision")

    assert response.status_code == 200
    data = response.json()

    assert "decision" in data
    assert "explanation" in data
    assert isinstance(data["explanation"], str)


def test_ops_copilot_report_contains_no_fake_gpu_claim():
    response = client.get("/ops/copilot")

    assert response.status_code == 200
    data = response.json()

    assert "observed_metrics" in data
    assert "bottlenecks" in data
