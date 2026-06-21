"""
FastAPI inference service.

  POST /api/rollout   { incline_deg, payload_kg, friction, num_slopes, speed_mps }
                      -> { params, terrain, frames }
  GET  /api/health

SECURITY: Nebius credentials (used by nebius.py to load policy checkpoints or
cache rollouts) are read from the environment / AWS profile at runtime and never
returned to the browser. The browser only ever receives rollout JSON.

Run (from the backend/ directory):
    uvicorn app:app --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import run_rollout


class RolloutParams(BaseModel):
    # Ranges double as validation and as the slider bounds documented for the UI.
    incline_deg: float = Field(0.0, ge=0.0, le=35.0, description="ground incline")
    payload_kg: float = Field(0.0, ge=0.0, le=30.0, description="carried weight")
    friction: float = Field(0.6, ge=0.05, le=1.5, description="ground friction")
    num_slopes: int = Field(0, ge=0, le=8, description="number of slopes")
    speed_mps: float = Field(1.0, ge=0.0, le=3.0, description="commanded speed")
    seconds: float = Field(8.0, ge=1.0, le=20.0, description="rollout duration")


app = FastAPI(title="Robot Rollout Inference")

# In dev the Vite proxy makes this same-origin; CORS is here so the UI can also
# hit the backend directly if you don't proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/rollout")
def rollout(params: RolloutParams):
    return run_rollout(params.model_dump())
