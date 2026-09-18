import threading
from typing import List

from core.telemetry.event import TelemetryEvent


class TelemetryCollector:
    """
    Thread-safe collector for inference telemetry events.
    """

    def __init__(self):
        self._events: List[TelemetryEvent] = []
        self._lock = threading.Lock()

    def record(self, event: TelemetryEvent) -> None:
        """Store a telemetry event."""

        with self._lock:
            self._events.append(event)

    def get_events(self) -> List[TelemetryEvent]:
        """Return a snapshot of all telemetry events."""

        with self._lock:
            return list(self._events)

    def count(self) -> int:
        """Return the number of recorded events."""

        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        """Remove all stored telemetry events."""

        with self._lock:
            self._events.clear()
