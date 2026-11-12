import math
import sys

import pytest

from ai.sla_predictor.predictor import SLAViolationPredictor


def test_low_sla_violation_risk():
    result = SLAViolationPredictor().predict(200.0, 50.0)

    assert result["risk_level"] == "low"
    assert result["predicted_violation"] is False
    assert result["risk_score"] == pytest.approx(0.25)
    assert result["risk_ratio"] == pytest.approx(0.25)


def test_medium_sla_violation_risk():
    result = SLAViolationPredictor().predict(100.0, 60.0)

    assert result["risk_level"] == "medium"
    assert result["predicted_violation"] is False
    assert result["risk_score"] == pytest.approx(0.60)


def test_high_sla_violation_risk():
    result = SLAViolationPredictor().predict(100.0, 90.0)

    assert result["risk_level"] == "high"
    assert result["predicted_violation"] is False
    assert result["risk_score"] == pytest.approx(0.90)


def test_critical_sla_violation_risk():
    result = SLAViolationPredictor().predict(100.0, 120.0)

    assert result["risk_level"] == "critical"
    assert result["predicted_violation"] is True
    assert result["risk_score"] == 1.0


def test_exact_medium_threshold_is_medium():
    result = SLAViolationPredictor().predict(100.0, 50.0)

    assert result["risk_level"] == "medium"


def test_exact_high_threshold_is_high():
    result = SLAViolationPredictor().predict(100.0, 80.0)

    assert result["risk_level"] == "high"


def test_exact_sla_budget_is_critical():
    result = SLAViolationPredictor().predict(100.0, 100.0)

    assert result["risk_level"] == "critical"
    assert result["predicted_violation"] is True
    assert result["risk_ratio"] == 1.0


@pytest.mark.parametrize("remaining", [-10.0, 0.0])
def test_expired_or_zero_remaining_sla_is_critical(remaining):
    result = SLAViolationPredictor().predict(remaining, 5.0)

    assert result["risk_level"] == "critical"
    assert result["predicted_violation"] is True
    assert result["risk_ratio"] == 1.0
    assert result["risk_score"] == 1.0


def test_zero_predicted_latency_has_low_risk():
    result = SLAViolationPredictor().predict(100.0, 0.0)

    assert result["risk_level"] == "low"
    assert result["risk_ratio"] == 0.0
    assert result["risk_score"] == 0.0
    assert result["predicted_violation"] is False


def test_missing_remaining_sla_is_rejected():
    with pytest.raises(ValueError):
        SLAViolationPredictor().predict(None, 10.0)


def test_missing_predicted_latency_is_rejected():
    with pytest.raises(ValueError):
        SLAViolationPredictor().predict(100.0, None)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf")],
)
def test_non_finite_remaining_sla_is_rejected(value):
    with pytest.raises(ValueError):
        SLAViolationPredictor().predict(value, 10.0)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf")],
)
def test_non_finite_predicted_latency_is_rejected(value):
    with pytest.raises(ValueError):
        SLAViolationPredictor().predict(100.0, value)


def test_negative_predicted_latency_is_rejected():
    with pytest.raises(ValueError):
        SLAViolationPredictor().predict(100.0, -1.0)


@pytest.mark.parametrize(
    "medium,high",
    [
        (0.9, 0.8),
        (0.0, 0.8),
        (0.5, 1.0),
        (-0.1, 0.8),
        (float("nan"), 0.8),
        (0.5, float("inf")),
    ],
)
def test_invalid_thresholds_are_rejected(medium, high):
    with pytest.raises(ValueError):
        SLAViolationPredictor(
            medium_threshold=medium,
            high_threshold=high,
        )


@pytest.mark.parametrize(
    "medium,high",
    [
        (None, 0.8),
        ("invalid", 0.8),
        (0.5, None),
        (0.5, "invalid"),
    ],
)
def test_non_numeric_thresholds_are_rejected(medium, high):
    with pytest.raises(ValueError):
        SLAViolationPredictor(
            medium_threshold=medium,
            high_threshold=high,
        )


@pytest.mark.parametrize(
    "remaining,predicted",
    [
        ("invalid", 10.0),
        (100.0, "invalid"),
    ],
)
def test_non_numeric_prediction_inputs_are_rejected(
    remaining,
    predicted,
):
    with pytest.raises(ValueError):
        SLAViolationPredictor().predict(remaining, predicted)


def test_custom_thresholds_are_respected():
    predictor = SLAViolationPredictor(
        medium_threshold=0.40,
        high_threshold=0.70,
    )

    result = predictor.predict(100.0, 50.0)

    assert result["risk_level"] == "medium"


def test_overflowing_risk_ratio_remains_finite():
    result = SLAViolationPredictor().predict(
        remaining_sla_ms=1e-300,
        predicted_latency_ms=1e300,
    )

    assert math.isfinite(result["risk_ratio"])
    assert result["risk_ratio"] == sys.float_info.max
    assert result["risk_score"] == 1.0
    assert result["risk_level"] == "critical"
    assert result["predicted_violation"] is True
