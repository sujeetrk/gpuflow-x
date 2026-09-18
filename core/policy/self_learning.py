from dataclasses import dataclass
import random
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SchedulingAction:
    batch_size: int
    batch_delay_ms: float


class SelfLearningPolicy:
    """
    Epsilon-greedy contextual bandit for GPUFlow-X.

    The policy maintains a value estimate for each
    (context, action) pair and updates that estimate
    from observed rewards.
    """

    def __init__(
        self,
        actions: List[SchedulingAction] | None = None,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
    ):
        if actions is None:
            actions = [
                SchedulingAction(1, 0),
                SchedulingAction(2, 0),
                SchedulingAction(4, 0),
                SchedulingAction(8, 0),
                SchedulingAction(1, 5),
                SchedulingAction(2, 5),
                SchedulingAction(4, 5),
                SchedulingAction(8, 5),
                SchedulingAction(1, 10),
                SchedulingAction(2, 10),
                SchedulingAction(4, 10),
                SchedulingAction(8, 10),
                SchedulingAction(1, 20),
                SchedulingAction(2, 20),
                SchedulingAction(4, 20),
                SchedulingAction(8, 20),
            ]

        if not actions:
            raise ValueError("At least one scheduling action is required.")

        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1.")

        if not 0 <= min_epsilon <= 1:
            raise ValueError("min_epsilon must be between 0 and 1.")

        if epsilon_decay <= 0 or epsilon_decay > 1:
            raise ValueError(
                "epsilon_decay must be greater than 0 and at most 1."
            )

        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.values: Dict[Tuple[Tuple[int, ...], SchedulingAction], float] = {}
        self.counts: Dict[Tuple[Tuple[int, ...], SchedulingAction], int] = {}

        self.total_updates = 0

    def _context_key(
        self,
        queue_length: int,
        predicted_latency_ms: float,
        current_batch_size: int,
    ) -> Tuple[int, ...]:
        """
        Convert continuous scheduler state into stable context buckets.
        """

        if queue_length <= 2:
            queue_bucket = 0
        elif queue_length <= 8:
            queue_bucket = 1
        elif queue_length <= 32:
            queue_bucket = 2
        else:
            queue_bucket = 3

        if predicted_latency_ms <= 5:
            latency_bucket = 0
        elif predicted_latency_ms <= 20:
            latency_bucket = 1
        elif predicted_latency_ms <= 50:
            latency_bucket = 2
        else:
            latency_bucket = 3

        if current_batch_size <= 1:
            batch_bucket = 0
        elif current_batch_size <= 4:
            batch_bucket = 1
        else:
            batch_bucket = 2

        return (
            queue_bucket,
            latency_bucket,
            batch_bucket,
        )

    def _key(
        self,
        context: Tuple[int, ...],
        action: SchedulingAction,
    ):
        return context, action

    def select_action(
        self,
        queue_length: int,
        predicted_latency_ms: float,
        current_batch_size: int,
    ) -> SchedulingAction:

        context = self._context_key(
            queue_length=queue_length,
            predicted_latency_ms=predicted_latency_ms,
            current_batch_size=current_batch_size,
        )

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation
        action_values = [
            (
                self.values.get(
                    self._key(context, action),
                    0.0,
                ),
                action,
            )
            for action in self.actions
        ]

        max_value = max(
            value
            for value, _ in action_values
        )

        best_actions = [
            action
            for value, action in action_values
            if value == max_value
        ]

        return random.choice(best_actions)

    def update(
        self,
        queue_length: int,
        predicted_latency_ms: float,
        current_batch_size: int,
        action: SchedulingAction,
        reward: float,
    ) -> None:

        context = self._context_key(
            queue_length=queue_length,
            predicted_latency_ms=predicted_latency_ms,
            current_batch_size=current_batch_size,
        )

        key = self._key(
            context,
            action,
        )

        old_value = self.values.get(
            key,
            0.0,
        )

        count = self.counts.get(
            key,
            0,
        ) + 1

        # Incremental mean reward update.
        new_value = old_value + (
            reward - old_value
        ) / count

        self.values[key] = new_value
        self.counts[key] = count

        self.total_updates += 1

        # Gradually reduce exploration.
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def action_value(
        self,
        queue_length: int,
        predicted_latency_ms: float,
        current_batch_size: int,
        action: SchedulingAction,
    ) -> float:

        context = self._context_key(
            queue_length=queue_length,
            predicted_latency_ms=predicted_latency_ms,
            current_batch_size=current_batch_size,
        )

        return self.values.get(
            self._key(context, action),
            0.0,
        )

    def statistics(self) -> dict:
        return {
            "actions": len(self.actions),
            "learned_state_action_pairs": len(self.values),
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
        }
