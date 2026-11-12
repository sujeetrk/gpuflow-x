import math
import sys


class SLAViolationPredictor:
    """
    Deterministic SLA violation risk estimator.

    Compares predicted inference latency with the remaining SLA budget.
    This is a rule-based estimator, not a trained ML classifier.
    """

    def __init__(
        self,
        medium_threshold=0.50,
        high_threshold=0.80,
    ):
        try:
            medium = float(medium_threshold)
            high = float(high_threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "SLA risk thresholds must be finite numbers."
            ) from exc

        if (
            not math.isfinite(medium)
            or not math.isfinite(high)
            or not 0 < medium < high < 1
        ):
            raise ValueError(
                "Thresholds must satisfy 0 < medium < high < 1."
            )

        self.medium_threshold = medium
        self.high_threshold = high

    def predict(self, remaining_sla_ms, predicted_latency_ms):
        """
        Estimate SLA violation risk.

        Risk levels:
            low:      ratio < medium_threshold
            medium:   medium_threshold <= ratio < high_threshold
            high:     high_threshold <= ratio < 1
            critical: ratio >= 1 or SLA already expired

        risk_score is capped at 1.0. risk_ratio retains the uncapped
        ratio, except that floating-point overflow is capped at the
        largest finite float.
        """
        if remaining_sla_ms is None:
            raise ValueError("remaining_sla_ms cannot be None.")

        if predicted_latency_ms is None:
            raise ValueError("predicted_latency_ms cannot be None.")

        try:
            remaining = float(remaining_sla_ms)
            predicted = float(predicted_latency_ms)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "SLA values must be valid numbers."
            ) from exc

        if not math.isfinite(remaining):
            raise ValueError("remaining_sla_ms must be finite.")

        if not math.isfinite(predicted):
            raise ValueError("predicted_latency_ms must be finite.")

        if predicted < 0:
            raise ValueError(
                "predicted_latency_ms cannot be negative."
            )

        # An expired or exhausted SLA is always critical.
        if remaining <= 0:
            ratio = 1.0
            risk_score = 1.0
            risk_level = "critical"
            predicted_violation = True

        else:
            ratio = predicted / remaining

            # Prevent infinity if division overflows.
            if not math.isfinite(ratio):
                ratio = sys.float_info.max

            risk_score = min(ratio, 1.0)

            if ratio >= 1.0:
                risk_level = "critical"
            elif ratio >= self.high_threshold:
                risk_level = "high"
            elif ratio >= self.medium_threshold:
                risk_level = "medium"
            else:
                risk_level = "low"

            predicted_violation = ratio >= 1.0

        return {
            "remaining_sla_ms": remaining,
            "predicted_latency_ms": predicted,
            "risk_ratio": ratio,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "predicted_violation": predicted_violation,
        }
