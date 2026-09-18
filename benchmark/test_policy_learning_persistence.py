import tempfile

from core.policy.learning_agent import LearningAgent
from core.policy.policy_storage import PolicyStorage
from core.policy.self_learning import SchedulingAction


def main():

    with tempfile.TemporaryDirectory() as temp_dir:

        policy_path = f"{temp_dir}/learned_policy.json"

        agent = LearningAgent()

        action = SchedulingAction(
            batch_size=4,
            batch_delay_ms=10,
        )

        print("=== GPUFlow-X Policy Persistence Test ===")

        for _ in range(10):

            agent.learn(
                queue_length=32,
                predicted_latency_ms=10.0,
                current_batch_size=4,
                action=action,
                average_latency_ms=2.0,
                average_queue_time_ms=1.0,
                throughput_requests_per_sec=5000.0,
            )

        before = agent.policy.action_value(
            queue_length=32,
            predicted_latency_ms=10.0,
            current_batch_size=4,
            action=action,
        )

        storage = PolicyStorage(
            policy_path
        )

        storage.save(
            agent.policy
        )

        restored_agent = LearningAgent()

        loaded = storage.load(
            restored_agent.policy
        )

        after = restored_agent.policy.action_value(
            queue_length=32,
            predicted_latency_ms=10.0,
            current_batch_size=4,
            action=action,
        )

        print("Policy saved:", True)
        print("Policy loaded:", loaded)
        print("Original value:", before)
        print("Restored value:", after)
        print(
            "Updates restored:",
            restored_agent.policy.total_updates,
        )

        assert loaded is True
        assert before == after
        assert (
            restored_agent.policy.total_updates
            == 10
        )

        print("Persistence test: PASSED")


if __name__ == "__main__":
    main()
