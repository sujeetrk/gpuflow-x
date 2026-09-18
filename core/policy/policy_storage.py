import json
from pathlib import Path

from core.policy.self_learning import (
    SchedulingAction,
    SelfLearningPolicy,
)


class PolicyStorage:

    def __init__(
        self,
        path="core/policy/learned_policy.json",
    ):
        self.path = Path(path)

    def save(
        self,
        policy: SelfLearningPolicy,
    ) -> None:

        data = {
            "epsilon": policy.epsilon,
            "epsilon_decay": policy.epsilon_decay,
            "min_epsilon": policy.min_epsilon,
            "total_updates": policy.total_updates,
            "values": [],
            "counts": [],
        }

        for (
            context,
            action,
        ), value in policy.values.items():

            data["values"].append(
                {
                    "context": list(context),
                    "batch_size": action.batch_size,
                    "batch_delay_ms": action.batch_delay_ms,
                    "value": value,
                }
            )

        for (
            context,
            action,
        ), count in policy.counts.items():

            data["counts"].append(
                {
                    "context": list(context),
                    "batch_size": action.batch_size,
                    "batch_delay_ms": action.batch_delay_ms,
                    "count": count,
                }
            )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.path,
            "w",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
            )

    def load(
        self,
        policy: SelfLearningPolicy,
    ) -> bool:

        if not self.path.exists():
            return False

        with open(
            self.path,
            "r",
        ) as file:

            data = json.load(file)

        policy.epsilon = data.get(
            "epsilon",
            policy.epsilon,
        )

        policy.total_updates = data.get(
            "total_updates",
            0,
        )

        policy.values.clear()
        policy.counts.clear()

        for item in data.get(
            "values",
            [],
        ):

            context = tuple(
                item["context"]
            )

            action = SchedulingAction(
                batch_size=item["batch_size"],
                batch_delay_ms=item["batch_delay_ms"],
            )

            policy.values[
                (context, action)
            ] = float(
                item["value"]
            )

        for item in data.get(
            "counts",
            [],
        ):

            context = tuple(
                item["context"]
            )

            action = SchedulingAction(
                batch_size=item["batch_size"],
                batch_delay_ms=item["batch_delay_ms"],
            )

            policy.counts[
                (context, action)
            ] = int(
                item["count"]
            )

        return True
