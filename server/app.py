"""
FastAPI application for the Cloud SRE OpenEnv Environment.

This module creates a stateful HTTP server that exposes CloudSREEnvironment
over HTTP endpoints. The server maintains episode state across reset/step
calls, enabling multi-step diagnostic episodes.

Note on design: Unlike single-step environments (e.g., reasoning_gym),
SRE incident diagnosis requires multi-step interaction. This server uses
server-side session state to persist the environment across HTTP calls,
while still complying with the OpenEnv Action/Observation/Environment spec.

Endpoints:
    POST /reset   — Start a new incident episode (task_id=1,2,3)
    POST /step    — Execute one SRE action, receive observation + reward
    GET  /state   — Inspect current episode state
    GET  /schema  — JSON schemas for Action and Observation
    GET  /        — War Room live monitoring dashboard
    GET  /health  — Health check

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from .environment import CloudSREEnvironment
    from .models import CloudSREAction, CloudSREObservation
except ImportError:
    from environment import CloudSREEnvironment  # type: ignore
    from models import CloudSREAction, CloudSREObservation  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cloud SRE OpenEnv",
    description=(
        "An SRE / DevOps environment where an AI agent diagnoses and resolves "
        "real-world infrastructure incidents across a simulated microservices cluster. "
        "\n\n## How to use\n"
        "1. POST `/reset` with `{\"task_id\": 1}` to start an incident episode\n"
        "2. POST `/step` with an action to advance the episode\n"
        "3. Repeat until `done=true` in the response\n"
        "4. GET `/` for the War Room live monitoring dashboard"
    ),
    version="1.0.0",
    contact={"name": "OpenEnv Cloud SRE"},
)

# Global environment instance — single session HTTP model
_env: Optional[CloudSREEnvironment] = CloudSREEnvironment()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1
    episode_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
async def reset(req: ResetRequest = ResetRequest()) -> ResetResponse:
    """Reset the environment to a new incident task.

    Args:
        task_id: 1=Alert Triage (Easy), 2=Bad Deployment (Med), 3=DB Deadlock (Hard)

    Returns:
        Initial observation with the active alert and task description.
    """
    global _env
    obs: CloudSREObservation = _env.reset(
        task_id=req.task_id,
        episode_id=req.episode_id,
    )
    return ResetResponse(
        observation=obs.model_dump(),
        reward=obs.reward,
        done=obs.done,
        info={"task_id": req.task_id, "episode_id": _env._episode_id},
    )


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
async def step(action: CloudSREAction) -> StepResponse:
    """Execute one SRE action and advance the episode.

    Args:
        action: A CloudSREAction object (see /schema for full spec).

    Returns:
        Observation, reward (0.0–1.0), and done flag.

    Raises:
        400: If the environment has not been reset yet.
    """
    global _env
    if _env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    obs: CloudSREObservation = _env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=obs.reward,
        done=obs.done,
        info={"step_count": obs.step_count, "cumulative_score": obs.cumulative_score},
    )


@app.get("/state", tags=["State Management"])
async def state() -> Dict[str, Any]:
    """Get the current environment state without advancing the episode."""
    global _env
    if _env._sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    sim = _env._sim
    return {
        "task_id": sim.task_level,
        "step_count": _env._step_count,
        "score": sim.score,
        "resolved": sim.resolved,
        "active_alerts": sim.alerts,
        "live_metrics": sim.live_metrics,
        "services_status": sim.services_status,
        "grader_state": sim.grader_state,
    }


@app.get("/schema", tags=["Schema"])
async def schema() -> Dict[str, Any]:
    """Get JSON schemas for CloudSREAction and CloudSREObservation."""
    return {
        "action": CloudSREAction.model_json_schema(),
        "observation": CloudSREObservation.model_json_schema(),
    }


@app.get("/health", tags=["Health"])
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "env": "cloud-sre-env", "version": "1.0.0"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def war_room_dashboard() -> HTMLResponse:
    """Serve the War Room live monitoring dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Cloud SRE War Room</h1><p>Dashboard not found.</p>")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 7860):
    """Run the Cloud SRE environment server."""
    import uvicorn
    port = int(os.getenv("PORT", str(port)))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
