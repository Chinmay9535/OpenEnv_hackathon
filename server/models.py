"""
Data models for the Cloud SRE OpenEnv Environment.

CloudSREObservation uses plain BaseModel (NOT openenv.core.Observation) to
avoid field conflicts with the base class's `reward: Optional[float] = None`
default, which the validator reads as 0.0 (out of range).

All reward values are strictly in (0.001, 0.981) — never exactly 0 or 1.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action
except ImportError:
    Action = BaseModel  # type: ignore


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class CloudSREAction(Action):
    """An action taken by the SRE agent in the Cloud SRE environment."""

    action_type: Literal[
        "query_metrics",
        "fetch_logs",
        "list_deployments",
        "rollback_deployment",
        "run_db_query",
        "resolve_incident",
        "noop",
    ] = Field(..., description="The type of SRE action to perform.")

    service: Optional[str] = Field(
        None,
        description=(
            "Target microservice (e.g. 'cache', 'payment-gateway', 'cart-service', "
            "'frontend', 'auth-service', 'order-service', 'api-gateway')."
        ),
    )
    metric: Optional[str] = Field(
        None,
        description="Metric to query: 'memory_usage', 'cpu_usage', 'error_rate', 'latency_p99'.",
    )
    lines: Optional[int] = Field(
        20,
        description="Number of recent log lines to tail. Default: 20.",
    )
    version: Optional[str] = Field(
        None,
        description="Deployment version to rollback to (e.g. 'v1.0.3', 'v2.3.0', 'v3.1.9').",
    )
    query: Optional[str] = Field(
        None,
        description="SQL or DB management query to execute against the cluster database.",
    )
    resolution_notes: Optional[str] = Field(
        None,
        description=(
            "Detailed notes describing the identified root cause and the fix applied. "
            "Be specific — mention the service, version, or PID involved."
        ),
    )


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class CloudSREObservation(BaseModel):
    """
    Observation returned to the agent after each environment step.

    Uses plain BaseModel (not openenv.core.Observation) to avoid field
    conflicts. All reward values are guaranteed strictly in (0.001, 0.981).
    """

    # Core RL fields — always set explicitly
    reward: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Cumulative grader reward for this episode, strictly in (0, 1).",
    )
    done: bool = Field(
        default=False,
        description="True when the incident is resolved and the episode ends.",
    )

    # Incident context
    active_alerts: List[str] = Field(
        default_factory=list,
        description="Currently active PagerDuty-style alerts (empty when resolved).",
    )
    task_description: str = Field(
        default="",
        description="Natural-language description of the current incident and goal.",
    )
    last_action_output: str = Field(
        default="",
        description="Raw text/JSON output from the previously executed SRE command.",
    )

    # Agent state
    step_count: int = Field(
        default=0,
        description="Number of steps taken in the current episode.",
    )
    cumulative_score: float = Field(
        default=0.001,
        description="Running sum of diagnostic rubric points earned so far.",
    )

    # Live telemetry
    live_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Real-time system telemetry: cpu (%), memory (%), error_rate (%), "
            "latency_p99_ms."
        ),
    )
    services_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Health map per microservice: 'healthy', 'degraded', or 'critical'.",
    )

    # Discovery aids — help zero-shot agents understand available actions
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "query_metrics",
            "fetch_logs",
            "list_deployments",
            "rollback_deployment",
            "run_db_query",
            "resolve_incident",
        ],
        description="Exhaustive list of valid action_type values for this environment.",
    )
    topology_hint: str = Field(
        default=(
            "Services: api-gateway → frontend → cart-service → database; "
            "auth-service validates all requests; "
            "payment-gateway → database; "
            "cache is standalone."
        ),
        description="High-level microservice dependency map to aid root-cause tracing.",
    )
