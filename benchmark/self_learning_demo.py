import time

from core.policy.learning_agent import LearningAgent
from core.policy.reward import SchedulerReward
from core.policy.self_learning import SchedulingAction
from core.scheduler.batch_scheduler import DynamicBatchScheduler
from core.batching.batch_config import BatchConfig
from inference.service import InferenceService


def run_action(action, request_count=32):

    scheduler = DynamicBatchScheduler(
        inference_service=InferenceService(),
        batch_config=BatchConfig(
            max_batch_size=action.batch_size,
            max_batch_delay_ms=action.batch_delay_ms,
        ),
        max_queue_size=request_count + 10,
    )

    scheduler.start()

    requests = [
        __import__(
            "core.queue.request",
            fromlist=["InferenceRequest"]
        ).InferenceRequest(
            [i] * 10
        )
        for i in range(request_count)
    ]

    start = time.perf_counter()

    for request in requests:
        scheduler.submit(request)

    deadline = time.perf_counter() + 10

    while scheduler.metrics()["requests_completed"] < request_count:
        if time.perf_counter() >= deadline:
            break

        time.sleep(0.005)

    runtime_ms = (
        time.perf_counter() - start
    ) * 1000

    metrics = scheduler.metrics()
    events = scheduler.telemetry.get_events()

    scheduler.stop()

    completed = [
        event
        for event in events
        if event.status == "completed"
    ]

    if not completed:
        return None

    latencies = [
        event.total_latency_ms
        for event in completed
        if event.total_latency_ms is not None
    ]

    queue_times = [
        event.queue_time_ms
        for event in completed
        if event.queue_time_ms is not None
    ]

    average_latency = (
        sum(latencies) / len(latencies)
    )

    average_queue = (
        sum(queue_times) / len(queue_times)
    )

    throughput = (
        len(completed)
        / (runtime_ms / 1000)
    )

    return {
        "average_latency_ms": average_latency,
        "average_queue_time_ms": average_queue,
        "throughput": throughput,
        "completed": metrics["requests_completed"],
    }


def main():

    agent = LearningAgent(
        reward_model=SchedulerReward()
    )

    queue_length = 32
    predicted_latency = 10.0
    current_batch_size = 4

    print("=== GPUFlow-X Self-Learning Demo ===")
    print()

    for episode in range(5):

        action = agent.choose_action(
            queue_length=queue_length,
            predicted_latency_ms=predicted_latency,
            current_batch_size=current_batch_size,
        )

        result = run_action(action)

        if result is None:
            print(
                "Episode",
                episode + 1,
                "failed."
            )
            continue

        reward = agent.learn(
            queue_length=queue_length,
            predicted_latency_ms=predicted_latency,
            current_batch_size=current_batch_size,
            action=action,
            average_latency_ms=result[
                "average_latency_ms"
            ],
            average_queue_time_ms=result[
                "average_queue_time_ms"
            ],
            throughput_requests_per_sec=result[
                "throughput"
            ],
        )

        print(
            f"Episode {episode + 1}: "
            f"batch={action.batch_size}, "
            f"delay={action.batch_delay_ms}ms, "
            f"latency={result['average_latency_ms']:.3f}ms, "
            f"queue={result['average_queue_time_ms']:.3f}ms, "
            f"throughput={result['throughput']:.2f} req/s, "
            f"reward={reward:.4f}"
        )

    print()
    print("Learning statistics:")
    print(agent.statistics())


if __name__ == "__main__":
    main()
