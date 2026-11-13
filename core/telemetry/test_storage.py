from core.telemetry.collector import TelemetryCollector
from core.telemetry.event import TelemetryEvent
from core.telemetry.storage import TelemetryStorage


def make_event(request_id):
    return TelemetryEvent(
        request_id=request_id,
        arrival_time=100.0,
        enqueue_time=100.1,
        start_time=100.2,
        end_time=100.5,
        queue_time_ms=100.0,
        inference_time_ms=200.0,
        total_latency_ms=300.0,
        batch_size=2,
        status="completed",
        device="cpu",
    )


def test_event_survives_collector_restart(tmp_path):
    database = tmp_path / "telemetry.db"

    first = TelemetryCollector(TelemetryStorage(database))
    first.record(make_event("request-1"))

    restored = TelemetryCollector(TelemetryStorage(database))

    assert restored.count() == 1
    assert restored.get_events()[0].request_id == "request-1"
    assert restored.get_events()[0].device == "cpu"


def test_duplicate_request_id_updates_existing_event(tmp_path):
    storage = TelemetryStorage(tmp_path / "telemetry.db")
    collector = TelemetryCollector(storage)

    collector.record(make_event("request-1"))

    updated = make_event("request-1")
    updated.device = "cuda:0"
    collector.record(updated)

    assert collector.count() == 1
    assert collector.get_events()[0].device == "cuda:0"
    assert storage.count() == 1


def test_multiple_events_are_persisted(tmp_path):
    collector = TelemetryCollector(
        TelemetryStorage(tmp_path / "telemetry.db")
    )

    collector.record(make_event("request-1"))
    collector.record(make_event("request-2"))

    restored = TelemetryCollector(
        TelemetryStorage(tmp_path / "telemetry.db")
    )

    assert restored.count() == 2


def test_clear_removes_memory_and_database_records(tmp_path):
    storage = TelemetryStorage(tmp_path / "telemetry.db")
    collector = TelemetryCollector(storage)

    collector.record(make_event("request-1"))
    collector.clear()

    assert collector.count() == 0
    assert storage.count() == 0
    assert TelemetryCollector(
        TelemetryStorage(tmp_path / "telemetry.db")
    ).count() == 0
