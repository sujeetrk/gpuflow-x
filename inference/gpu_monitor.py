"""Hardware health monitoring for GPUFlow-X."""

import torch


class GPUHealthMonitor:
    """Collect GPU health and device fault information."""

    def __init__(self, device_manager):
        self.device_manager = device_manager

    def snapshot(self):
        devices = self.device_manager.devices
        metrics = self.device_manager.metrics()
        cuda_available = bool(torch.cuda.is_available())
        unhealthy = metrics.get("unhealthy_devices", {})

        result = {
            "status": "healthy",
            "cuda_available": cuda_available,
            "gpu_count": len(devices),
            "total_gpus": metrics.get("total_gpus", len(devices)),
            "available_gpus": metrics.get("available_gpus", len(devices)),
            "cpu_fallback": metrics["cpu_fallback"],
            "active_assignments": metrics["active_assignments"],
            "device_loads": metrics["device_loads"],
            "unhealthy_devices": unhealthy,
            "failure_counts": metrics.get("failure_counts", {}),
            "last_failure": metrics.get("last_failure", {}),
            "gpus": [],
        }

        if not cuda_available:
            result["status"] = "cpu_fallback"
            result["monitoring_note"] = (
                "CUDA is unavailable. Hardware GPU metrics "
                "cannot be collected in this environment."
            )
            return result

        if unhealthy:
            result["status"] = "degraded"

        for device in devices:
            gpu_info = {
                "device": device,
                "available": device not in unhealthy,
                "quarantined": device in unhealthy,
                "quarantine_reason": unhealthy.get(device),
                "utilization_percent": None,
                "memory_allocated_bytes": None,
                "memory_reserved_bytes": None,
                "memory_total_bytes": None,
                "memory_free_bytes": None,
                "monitoring_error": None,
            }

            if device in unhealthy:
                result["gpus"].append(gpu_info)
                continue

            try:
                index = int(device.split(":")[1])
                properties = torch.cuda.get_device_properties(index)

                gpu_info["name"] = properties.name
                gpu_info["memory_total_bytes"] = properties.total_memory
                gpu_info["memory_allocated_bytes"] = (
                    torch.cuda.memory_allocated(index)
                )
                gpu_info["memory_reserved_bytes"] = (
                    torch.cuda.memory_reserved(index)
                )

                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                gpu_info["memory_free_bytes"] = free_bytes
                gpu_info["memory_total_bytes"] = total_bytes

            except Exception as exc:
                gpu_info["available"] = False
                gpu_info["monitoring_error"] = str(exc)
                result["status"] = "degraded"

            result["gpus"].append(gpu_info)

        result["utilization_note"] = (
            "GPU utilization percentage is unavailable through "
            "the configured PyTorch monitoring interface."
        )
        return result
