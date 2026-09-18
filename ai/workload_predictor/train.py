import joblib
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(
    "ai/workload_predictor/workload_dataset.csv"
)

MODEL_PATH = Path(
    "ai/workload_predictor/workload_predictor.joblib"
)

METRICS_PATH = Path(
    "ai/workload_predictor/workload_predictor_metrics.json"
)


FEATURES = [
    "queue_time_ms",
    "inference_time_ms",
    "total_latency_ms",
    "batch_size",
]

TARGET = "next_latency_ms"


def train_model():
    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5
    r2 = r2_score(y_test, predictions)

    results = {
        "model": "RandomForestRegressor",
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "mae_ms": float(mae),
        "rmse_ms": float(rmse),
        "r2": float(r2),
        "features": FEATURES,
        "target": TARGET,
    }

    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
    	json.dumps(results, indent=4)
    )

    print("=== GPUFlow-X Workload Predictor ===")
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("MAE (ms):", mae)
    print("RMSE (ms):", rmse)
    print("R2:", r2)


if __name__ == "__main__":
    train_model()

print("Model saved to:", MODEL_PATH)
print("Metrics saved to:", METRICS_PATH)
