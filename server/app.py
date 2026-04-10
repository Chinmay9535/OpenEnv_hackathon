"""
Cloud SRE OpenEnv — app.py

Uses create_app() from openenv.core so the server gets the full framework:
  - POST /reset, POST /step, GET /state, GET /schema  (HTTP)
  - WebSocket /ws  ← the validator uses THIS to read task scores
  - GET / → War Room dashboard (added as custom route)

The validator connects via WebSocket (/ws), calls reset+step, reads
observation.reward from serialize_observation(). Without /ws the validator
gets None → 0.0 → "score out of range".
"""

import os
from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app

try:
    from .environment import CloudSREEnvironment
    from .models import CloudSREAction, CloudSREObservation
except ImportError:
    from environment import CloudSREEnvironment   # type: ignore
    from models import CloudSREAction, CloudSREObservation  # type: ignore

# create_app() builds the full OpenEnv FastAPI app including /ws WebSocket.
# env= must be a CALLABLE factory (not an instance) — called per WS session.
app = create_app(
    env=CloudSREEnvironment,          # factory: called once per WS connection
    action_cls=CloudSREAction,
    observation_cls=CloudSREObservation,
    env_name="cloud-sre-env",
)

# ── War Room dashboard preserved as custom route ──────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def war_room_dashboard() -> HTMLResponse:
    """Live War Room monitoring dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Cloud SRE War Room</h1><p>Dashboard coming soon.</p>")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    port = int(os.getenv("PORT", str(port)))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
