import joblib
import pandas as pd
from pathlib import Path


class LatencyPredictor:

    FEATURES = [
        "configured_batch_size",
        "batch_delay_ms",
        "request_count",
    ]

    def __init__(
        self,
        model_path="ai/latency_predictor/latency_predictor.joblib",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    def predict(
        self,
        batch_size: int,
        batch_delay_ms: float,
        request_count: int,
    ) -> float:

        features = pd.DataFrame(
            [
                {
                    "configured_batch_size": batch_size,
                    "batch_delay_ms": batch_delay_ms,
                    "request_count": request_count,
                }
            ],
            columns=self.FEATURES,
        )

        prediction = self.model.predict(features)

        return float(prediction[0])
