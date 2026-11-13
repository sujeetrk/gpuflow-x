"""SQLite-backed durable storage for inference telemetry."""

import json
import sqlite3
import threading
from dataclasses import asdict

from core.telemetry.event import TelemetryEvent


class TelemetryStorage:
    """Persist telemetry events safely across application restarts."""

    def __init__(self, path="data/gpuflow_telemetry.db"):
        self.path = str(path)
        self._lock = threading.RLock()

        if self.path != ":memory:":
            from pathlib import Path
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(
            self.path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._lock, self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    request_id TEXT PRIMARY KEY,
                    event_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_status
                ON telemetry_events (
                    json_extract(event_json, '$.status')
                )
            """)

    def save(self, event):
        """Insert or update one event by request ID."""
        payload = json.dumps(asdict(event))

        with self._lock, self._connect() as connection:
            connection.execute("""
                INSERT INTO telemetry_events (request_id, event_json)
                VALUES (?, ?)
                ON CONFLICT(request_id)
                DO UPDATE SET
                    event_json = excluded.event_json,
                    recorded_at = CURRENT_TIMESTAMP
            """, (event.request_id, payload))

    def load_all(self):
        """Load all persisted events in insertion order."""
        with self._lock, self._connect() as connection:
            rows = connection.execute("""
                SELECT event_json
                FROM telemetry_events
                ORDER BY rowid
            """).fetchall()

        return [
            TelemetryEvent(**json.loads(row["event_json"]))
            for row in rows
        ]

    def get_by_request_id(self, request_id):
        """Return one persisted event by request ID."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT event_json FROM telemetry_events WHERE request_id = ?",
                (request_id,),
            ).fetchone()

        if row is None:
            return None

        return TelemetryEvent(**json.loads(row["event_json"]))

    def count(self):
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM telemetry_events"
            ).fetchone()
            return row["total"]

    def clear(self):
        """Delete all persisted telemetry records."""
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM telemetry_events")
