"""Thread-safe in-memory and optional persistent telemetry collector."""

import threading
from typing import List

from core.telemetry.event import TelemetryEvent


class TelemetryCollector:
    """Collect telemetry events with optional SQLite persistence."""

    def __init__(self, storage=None):
        self._events: List[TelemetryEvent] = []
        self._lock = threading.RLock()
        self.storage = storage

        if self.storage is not None:
            self._events = self.storage.load_all()

    def record(self, event: TelemetryEvent) -> None:
        """Store an event in memory and, when configured, in SQLite."""
        with self._lock:
            if self.storage is not None:
                self.storage.save(event)

            for index, existing in enumerate(self._events):
                if existing.request_id == event.request_id:
                    self._events[index] = event
                    break
            else:
                self._events.append(event)

    def get_event(self, request_id):
        """Find an event in memory or persistent storage."""
        with self._lock:
            for event in self._events:
                if event.request_id == request_id:
                    return event

            if self.storage is not None:
                event = self.storage.get_by_request_id(request_id)
                if event is not None:
                    self._events.append(event)
                return event

            return None

    def get_events(self) -> List[TelemetryEvent]:
        """Return a snapshot of recorded events."""
        with self._lock:
            return list(self._events)

    def count(self) -> int:
        """Return the number of unique recorded request events."""
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        """Clear in-memory and persisted telemetry."""
        with self._lock:
            self._events.clear()

            if self.storage is not None:
                self.storage.clear()
