"""GPU device discovery, placement, load balancing, and fault tracking."""

import threading

import torch


class GPUDeviceManager:
    """Select the least-loaded healthy GPU with safe CPU fallback."""

    def __init__(self, device_count=None):
        if device_count is None:
            device_count = (
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0
            )

        if device_count < 0:
            raise ValueError("device_count cannot be negative")

        self._devices = [
            f"cuda:{i}" for i in range(device_count)
        ]
        self._loads = {device: 0 for device in self._devices}
        self._unhealthy = {}
        self._failure_counts = {
            device: 0 for device in self._devices
        }
        self._last_failure = {}
        self._lock = threading.Lock()
        self._next_index = 0

    @property
    def devices(self):
        return list(self._devices)

    def select_device(self):
        """Choose the least-loaded healthy GPU; CPU if none remain."""
        with self._lock:
            healthy = [
                device for device in self._devices
                if device not in self._unhealthy
            ]

            if not healthy:
                return "cpu"

            minimum = min(self._loads[device] for device in healthy)
            candidates = [
                i for i, device in enumerate(self._devices)
                if (
                    device in healthy
                    and self._loads[device] == minimum
                )
            ]

            selected_index = next(
                (
                    i for i in candidates
                    if i >= self._next_index
                ),
                candidates[0],
            )

            device = self._devices[selected_index]
            self._loads[device] += 1
            self._next_index = (
                selected_index + 1
            ) % len(self._devices)

            return device

    def mark_unhealthy(self, device, reason):
        """Quarantine a failed GPU until it is explicitly recovered."""
        with self._lock:
            if device not in self._loads:
                raise ValueError(f"Unknown device: {device}")

            self._failure_counts[device] += 1
            self._unhealthy[device] = str(reason)
            self._last_failure[device] = str(reason)

    def mark_healthy(self, device):
        """Remove a GPU from quarantine after an external health check."""
        with self._lock:
            if device not in self._loads:
                raise ValueError(f"Unknown device: {device}")

            self._unhealthy.pop(device, None)

    def release_device(self, device):
        """Release one active assignment."""
        if device == "cpu":
            return

        with self._lock:
            if device not in self._loads:
                raise ValueError(f"Unknown device: {device}")

            if self._loads[device] == 0:
                raise ValueError(
                    f"No active assignment for {device}"
                )

            self._loads[device] -= 1

    def load_snapshot(self):
        """Return a thread-safe copy of active GPU assignments."""
        with self._lock:
            return dict(self._loads)

    def metrics(self):
        """Return device availability, assignment, and fault metrics."""
        with self._lock:
            healthy_devices = [
                device for device in self._devices
                if device not in self._unhealthy
            ]

            return {
                "available_gpus": len(healthy_devices),
                "total_gpus": len(self._devices),
                "unhealthy_devices": dict(self._unhealthy),
                "failure_counts": dict(self._failure_counts),
                "last_failure": dict(self._last_failure),
                "active_assignments": sum(self._loads.values()),
                "device_loads": dict(self._loads),
                "cpu_fallback": not bool(healthy_devices),
            }
