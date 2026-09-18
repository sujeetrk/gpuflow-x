import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(
    "ai/latency_predictor/latency_dataset.csv"
)

MODEL_PATH = Path(
    "ai/latency_predictor/latency_predictor.joblib"
)

METRICS_PATH = Path(
    "ai/latency_predictor/latency_predictor_metrics.json"
)


FEATURES = [
    "configured_batch_size",
    "batch_delay_ms",
    "request_count",
]

TARGET = "average_latency_ms"


def main():

    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        max_depth=8,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    metrics = {
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "mae_ms": float(mae),
        "rmse_ms": float(rmse),
        "r2": float(r2),
        "features": FEATURES,
        "target": TARGET,
    }

    with open(
        METRICS_PATH,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("=== GPUFlow-X Latency Predictor ===")
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("MAE (ms):", mae)
    print("RMSE (ms):", rmse)
    print("R²:", r2)
    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)


if __name__ == "__main__":
    main()
