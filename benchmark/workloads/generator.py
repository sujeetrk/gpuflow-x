import random
import time
from typing import List


class WorkloadGenerator:
    def __init__(self, input_size: int = 10, seed: int = 42):
        self.input_size = input_size
        self.random = random.Random(seed)

    def generate_request(self) -> List[float]:
        return [
            self.random.uniform(-1.0, 1.0)
            for _ in range(self.input_size)
        ]

    def generate_requests(self, count: int) -> List[List[float]]:
        if count < 1:
            raise ValueError("Request count must be at least 1.")

        return [
            self.generate_request()
            for _ in range(count)
        ]

    def generate_constant_rate(
        self,
        count: int,
        requests_per_second: float
    ) -> List[List[float]]:
        if requests_per_second <= 0:
            raise ValueError("Requests per second must be greater than 0.")

        interval = 1.0 / requests_per_second
        requests = []

        for _ in range(count):
            requests.append(self.generate_request())
            time.sleep(interval)

        return requests
