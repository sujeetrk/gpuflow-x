from pathlib import Path

from ai.workload_predictor.predictor import WorkloadPredictor


MODEL_PATH = Path(
    "ai/workload_predictor/workload_predictor.joblib"
)


def test_predictor_loads_model():

    predictor = WorkloadPredictor(
        model_path=MODEL_PATH
    )

    assert predictor.model is not None


def test_predictor_returns_latency():

    predictor = WorkloadPredictor(
        model_path=MODEL_PATH
    )

    prediction = predictor.predict(
        queue_time_ms=25.0,
        inference_time_ms=0.1,
        total_latency_ms=25.1,
        batch_size=8,
    )

    assert isinstance(prediction, float)
    assert prediction >= 0.0


def test_predictor_handles_different_batch_sizes():

    predictor = WorkloadPredictor(
        model_path=MODEL_PATH
    )

    prediction_small = predictor.predict(
        queue_time_ms=10.0,
        inference_time_ms=0.1,
        total_latency_ms=10.1,
        batch_size=2,
    )

    prediction_large = predictor.predict(
        queue_time_ms=30.0,
        inference_time_ms=0.2,
        total_latency_ms=30.2,
        batch_size=8,
    )

    assert prediction_small >= 0.0
    assert prediction_large >= 0.0
