from ai.latency_predictor.predictor import LatencyPredictor


def test_predictor_loads_model():
    predictor = LatencyPredictor()

    assert predictor.model is not None


def test_predictor_returns_latency():
    predictor = LatencyPredictor()

    result = predictor.predict(
        batch_size=8,
        batch_delay_ms=10,
        request_count=32,
    )

    assert isinstance(result, float)
    assert result >= 0


def test_predictor_handles_different_batch_sizes():
    predictor = LatencyPredictor()

    predictions = []

    for batch_size in [1, 2, 4, 8]:

        prediction = predictor.predict(
            batch_size=batch_size,
            batch_delay_ms=10,
            request_count=32,
        )

        predictions.append(prediction)

    assert len(predictions) == 4
    assert all(
        isinstance(value, float)
        for value in predictions
    )
    assert all(
        value >= 0
        for value in predictions
    )
