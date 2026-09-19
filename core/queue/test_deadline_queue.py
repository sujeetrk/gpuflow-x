import time

from core.queue.request import InferenceRequest
from core.queue.request_queue import RequestQueue


def make_request(
    priority=0,
    deadline_time=None,
    sla_budget_ms=None,
):
    return InferenceRequest(
        values=[1.0] * 10,
        priority=priority,
        deadline_time=deadline_time,
        sla_budget_ms=sla_budget_ms,
    )


def test_earliest_deadline_first():
    queue = RequestQueue()

    now = time.perf_counter()

    late = make_request(deadline_time=now + 10)
    early = make_request(deadline_time=now + 2)
    middle = make_request(deadline_time=now + 5)

    queue.enqueue(late)
    queue.enqueue(early)
    queue.enqueue(middle)

    assert queue.dequeue() == early
    assert queue.dequeue() == middle
    assert queue.dequeue() == late


def test_higher_priority_first_for_equal_deadline():
    queue = RequestQueue()

    deadline = time.perf_counter() + 5

    low = make_request(priority=1, deadline_time=deadline)
    high = make_request(priority=10, deadline_time=deadline)
    medium = make_request(priority=5, deadline_time=deadline)

    queue.enqueue(low)
    queue.enqueue(high)
    queue.enqueue(medium)

    assert queue.dequeue() == high
    assert queue.dequeue() == medium
    assert queue.dequeue() == low


def test_deadline_requests_before_no_deadline():
    queue = RequestQueue()

    no_deadline = make_request(priority=100)
    with_deadline = make_request(
        deadline_time=time.perf_counter() + 10
    )

    queue.enqueue(no_deadline)
    queue.enqueue(with_deadline)

    assert queue.dequeue() == with_deadline
    assert queue.dequeue() == no_deadline


def test_priority_order_without_deadlines():
    queue = RequestQueue()

    low = make_request(priority=1)
    high = make_request(priority=10)
    medium = make_request(priority=5)

    queue.enqueue(low)
    queue.enqueue(high)
    queue.enqueue(medium)

    assert queue.dequeue() == high
    assert queue.dequeue() == medium
    assert queue.dequeue() == low


def test_sla_deadline_is_automatically_created():
    request = make_request(sla_budget_ms=500)

    assert request.deadline_time is not None

    expected_deadline = (
        request.arrival_time + 0.5
    )

    assert abs(
        request.deadline_time - expected_deadline
    ) < 0.001


def test_invalid_sla_budget_is_rejected():
    try:
        make_request(sla_budget_ms=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_deadline_missed_detection():
    request = make_request(
        deadline_time=time.perf_counter() - 1
    )

    assert request.deadline_missed is True


def test_remaining_sla_is_negative_after_deadline():
    request = make_request(
        deadline_time=time.perf_counter() - 1
    )

    assert request.remaining_sla_ms < 0
