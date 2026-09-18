from abc import ABC, abstractmethod
from typing import Any


class SchedulerPolicy(ABC):

    @abstractmethod
    def decide(self, state: Any) -> dict:
        """Return scheduling decisions based on the current system state."""
        pass
