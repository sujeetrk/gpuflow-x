"""Explain GPUFlow-X scheduler decisions using observed metrics."""

from statistics import mean


class AIOpsCopilot:
    DECISION_EXPLANATIONS = {
        "expired_deadline_drain":
            "A queued request deadline has expired. The scheduler "
            "bypasses normal policy and drains a maximum-sized batch "
            "without collection delay.",

        "urgent_deadline":
            "A request deadline is within the configured guard window. "
            "The scheduler uses a singleton batch with zero delay.",

        "backlog_throughput":
            "The queue reached the configured batch capacity. "
            "The scheduler bypasses a costly policy call and drains "
            "a full batch immediately.",

        "deadline_constrained":
            "The predicted collection delay was reduced to respect "
            "the earliest queued request's deadline.",

        "backlog":
            "The scheduler detected queued workload and selected "
            "a batching decision to help process it.",

        "default":
            "The scheduler used its regular policy to select "
            "the batch size and collection delay.",
    }

    def explain_decision(self, decision):
        if not decision:
            return "No scheduler decision is available yet."

        reason = decision.get("ai_reason", "default")
        explanation = self.DECISION_EXPLANATIONS.get(
            reason,
            self.DECISION_EXPLANATIONS["default"],
        )

        batch_size = decision.get("batch_size", "unknown")
        delay = decision.get("batch_delay_ms", "unknown")

        return (
            f"{explanation} "
            f"Final batch size: {batch_size}; "
            f"collection delay: {delay} ms. "
            f"Decision reason: {reason}."
        )

    @staticmethod
    def _event_value(event, key, default=None):
        if isinstance(event, dict):
            return event.get(key, default)
        return getattr(event, key, default)

    def analyze(self, metrics, telemetry_events=None):
        """Generate grounded operational insights from scheduler metrics."""
        telemetry_events = list(telemetry_events or [])
        completed = metrics.get("requests_completed", 0) or 0
        failed = metrics.get("requests_failed", 0) or 0
        submitted = metrics.get("requests_submitted", completed + failed) or 0
        queue_length = metrics.get("queue_length", 0) or 0
        deadline_misses = metrics.get("deadline_misses", 0) or 0

        bottlenecks = []
        recommendations = []

        if queue_length > 0:
            bottlenecks.append(
                f"Queue backlog detected: {queue_length} requests waiting."
            )
            recommendations.append(
                "Inspect batch size, collection delay, and inference "
                "throughput while the queue remains elevated."
            )

        failure_rate = failed / max(completed + failed, 1)

        if failed:
            bottlenecks.append(
                f"{failed} inference requests failed "
                f"({failure_rate:.1%} of completed or failed requests)."
            )
            recommendations.append(
                "Inspect failed-request telemetry and backend exceptions."
            )

        if deadline_misses:
            bottlenecks.append(
                f"{deadline_misses} deadline misses were recorded."
            )
            recommendations.append(
                "Review deadline pressure, queue wait time, "
                "and inference latency."
            )

        latencies = [
            self._event_value(event, "total_latency_ms")
            for event in telemetry_events
            if self._event_value(event, "total_latency_ms") is not None
            and self._event_value(event, "status") == "completed"
        ]

        queue_times = [
            self._event_value(event, "queue_time_ms")
            for event in telemetry_events
            if self._event_value(event, "queue_time_ms") is not None
            and self._event_value(event, "status") == "completed"
        ]

        inference_times = [
            self._event_value(event, "inference_time_ms")
            for event in telemetry_events
            if self._event_value(event, "inference_time_ms") is not None
            and self._event_value(event, "status") == "completed"
        ]

        avg_latency = mean(latencies) if latencies else None
        avg_queue = mean(queue_times) if queue_times else None
        avg_inference = mean(inference_times) if inference_times else None

        if avg_queue is not None and avg_inference is not None:
            if avg_queue > avg_inference:
                bottlenecks.append(
                    "Average queue wait exceeds average inference time."
                )
                recommendations.append(
                    "Investigate queue buildup and batching delay."
                )

        if avg_latency is not None and avg_latency > 0:
            if max(latencies) > 2 * avg_latency:
                bottlenecks.append(
                    "Some completed requests experienced latency "
                    "above twice the observed average."
                )
                recommendations.append(
                    "Inspect high-latency requests and workload variability."
                )

        decision = metrics.get("last_ai_decision") or {}

        if not bottlenecks:
            bottlenecks.append(
                "No bottleneck was identified from the supplied metrics."
            )

        if not recommendations:
            recommendations.append(
                "Continue monitoring queue length, latency, failures, "
                "and deadline misses."
            )

        return {
            "summary": (
                f"GPUFlow-X submitted {submitted} requests, completed "
                f"{completed}, and failed {failed}. "
                f"Current queue length: {queue_length}."
            ),
            "decision_explanation": self.explain_decision(decision),
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "observed_metrics": {
                "requests_submitted": submitted,
                "requests_completed": completed,
                "requests_failed": failed,
                "queue_length": queue_length,
                "deadline_misses": deadline_misses,
                "average_latency_ms": avg_latency,
                "average_queue_time_ms": avg_queue,
                "average_inference_time_ms": avg_inference,
            },
        }
