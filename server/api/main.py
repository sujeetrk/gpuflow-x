from fastapi import FastAPI

app = FastAPI(
    title="GPUFlow-X",
    description="AI-Driven Self-Learning GPU Inference Scheduler",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "project": "GPUFlow-X",
        "status": "running",
        "phase": "Phase 0 - Foundation",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
