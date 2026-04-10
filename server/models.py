"""
Data models for the Cloud SRE OpenEnv Environment.

IMPORTANT: CloudSREObservation intentionally inherits from plain BaseModel
(not openenv.core.Observation) to avoid field conflicts with the base class's
`reward: Optional[float] = None` and `metadata: dict` (required) fields.
The openenv.core.Observation base class has `reward` typed as
`bool | int | float | None` with default=None, which causes serialization
issues where the validator sees None/0 as out-of-range scores.

CloudSREAction still inherits from openenv.core.Action for schema compliance.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action
except ImportError:
    Action = BaseModel  # type: ignore


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
        description="Target microservice name (e.g., 'cache', 'payment-gateway', 'cart-service', 'frontend').",
    )
    metric: Optional[str] = Field(
        None,
        description="Metric to query (e.g., 'memory_usage', 'error_rate', 'cpu_usage', 'latency_p99').",
    )
    lines: Optional[int] = Field(
        20,
        description="Number of recent log lines to tail from the service.",
    )
    version: Optional[str] = Field(
        None,
        description="Deployment version to rollback to (e.g., 'v1.0.3').",
    )
    query: Optional[str] = Field(
        None,
        description="SQL or database management query to execute.",
    )
    resolution_notes: Optional[str] = Field(
        None,
        description="Detailed notes describing the identified root cause and the fix applied.",
    )


class CloudSREObservation(BaseModel):
    """Observation returned to the agent after each step.

    Uses plain BaseModel (not openenv.core.Observation) to avoid field
    conflicts from the base class's reward: Optional[float] = None default,
    which the validator would read as 0 (out of range).

    All reward values are guaranteed strictly in (0.001, 0.981).
    """

    # Core RL fields — always set explicitly, never use defaults
    reward: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Grader reward for this step, strictly in (0, 1).",
    )
    done: bool = Field(
        default=False,
        description="True when the incident has been resolved and the episode ends.",
    )

    # SRE-specific observation fields
    active_alerts: List[str] = Field(
        default_factory=list,
        description="List of currently active PagerDuty-style alerts.",
    )
    task_description: str = Field(
        default="",
        description="Natural-language description of the current incident.",
    )
    last_action_output: str = Field(
        default="",
        description="Raw text output from the previously executed SRE command.",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken in the current episode.",
    )
    cumulative_score: float = Field(
        default=0.001,
        description="Running grader score based on correct diagnostic steps.",
    )
    live_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Real-time telemetry: cpu, memory, error_rate.",
    )
    services_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Health status per microservice: 'healthy', 'degraded', 'critical'.",
    )
