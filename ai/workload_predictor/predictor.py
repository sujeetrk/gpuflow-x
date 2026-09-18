import joblib
import pandas as pd
from pathlib import Path


class WorkloadPredictor:

    FEATURES = [
        "queue_time_ms",
        "inference_time_ms",
        "total_latency_ms",
        "batch_size",
    ]

    def __init__(
        self,
        model_path="ai/workload_predictor/workload_predictor.joblib",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    def predict(
        self,
        queue_time_ms: float,
        inference_time_ms: float,
        total_latency_ms: float,
        batch_size: int,
    ) -> float:

        features = pd.DataFrame(
            [
                [
                    queue_time_ms,
                    inference_time_ms,
                    total_latency_ms,
                    batch_size,
                ]
            ],
            columns=self.FEATURES,
        )

        prediction = self.model.predict(features)

        return float(prediction[0])
