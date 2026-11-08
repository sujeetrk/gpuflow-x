import json
import time

from fastapi.testclient import TestClient

import server.api.main as api_module
from core.scheduler.ai_scheduler import AIScheduler
from inference.service import InferenceService


def test_infer_api_completes_learning_and_policy_persistence(
    tmp_path, monkeypatch
):
    policy_path = tmp_path / "integration-policy.json"

    isolated_scheduler = AIScheduler(
        inference_service=InferenceService(),
        max_queue_size=20,
        learning_enabled=True,
        policy_path=str(policy_path),
    )

    monkeypatch.setattr(
        api_module,
        "scheduler",
        isolated_scheduler,
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/infer",
            json={"values": [0.1] * 10},
        )

        assert response.status_code == 200, response.text
        request_id = response.json()["request_id"]

        deadline = time.monotonic() + 10
        result = None

        while time.monotonic() < deadline:
            status_response = client.get(
                f"/requests/{request_id}"
            )
            assert status_response.status_code == 200

            result = status_response.json()

            if result["status"] in ("completed", "failed"):
                break

            time.sleep(0.02)

        assert result is not None
        assert result["status"] == "completed", result
        assert "output" in result

        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics = metrics_response.json()

        assert metrics["requests_submitted"] == 1
        assert metrics["requests_completed"] == 1
        assert metrics["requests_failed"] == 0
        assert metrics["telemetry_events"] == 1
        assert metrics["last_ai_decision"] is not None
        assert metrics["last_reward"] is not None
        assert metrics["learning_statistics"]["decisions"] >= 1

        assert metrics["policy_save_successes"] >= 1, (
            "Policy persistence failed. "
            f"Save error: {metrics['last_policy_save_error']}"
        )

        assert metrics["policy_save_failures"] == 0
        assert metrics["last_policy_save_error"] is None

        assert policy_path.exists()

        saved_policy = json.loads(policy_path.read_text())
        assert saved_policy["total_updates"] >= 1
