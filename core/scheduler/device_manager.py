"""GPU device discovery, placement, and load balancing."""

import threading

import torch


class GPUDeviceManager:
    """Select the least-loaded GPU with safe CPU fallback."""

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
        self._lock = threading.Lock()
        self._next_index = 0

    @property
    def devices(self):
        return list(self._devices)

    def select_device(self):
        """Choose the least-loaded GPU; CPU if no GPU exists."""
        if not self._devices:
            return "cpu"

        with self._lock:
            minimum = min(self._loads.values())

            candidates = [
                i for i, device in enumerate(self._devices)
                if self._loads[device] == minimum
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
        """Return device counts and current assignment load."""
        with self._lock:
            return {
                "available_gpus": len(self._devices),
                "active_assignments": sum(self._loads.values()),
                "device_loads": dict(self._loads),
                "cpu_fallback": not bool(self._devices),
            }
