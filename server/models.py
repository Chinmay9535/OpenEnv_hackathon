"""
Data models for the Cloud SRE OpenEnv Environment.

The CloudSRE environment simulates a microservices infrastructure where
an AI agent acts as an on-call SRE, diagnosing and resolving incidents.

Action Space:
    CloudSREAction — defines all available SRE commands

Observation Space:
    CloudSREObservation — the agent's view of the current system state
"""

from typing import Dict, List, Literal, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from pydantic import BaseModel as Action, BaseModel as Observation  # type: ignore


class CloudSREAction(Action):
    """An action taken by the SRE agent in the Cloud SRE environment.

    The agent selects one action per step. The action_type determines
    which operation is executed against the simulated infrastructure.
    """

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
        description="SQL or database management query to execute (e.g., 'SELECT * FROM pg_stat_activity').",
    )
    resolution_notes: Optional[str] = Field(
        None,
        description="Detailed notes describing the identified root cause and the fix applied.",
    )


class CloudSREObservation(Observation):
    """Observation returned to the agent after each step in the Cloud SRE environment.

    The observation provides the agent with full situational awareness of the
    current incident, including firing alerts, the last command output, and
    live system telemetry for contextual reasoning.
    """

    # OpenEnv required fields — reward signal and episode termination flag
    reward: float = Field(
        default=0.0,
        description="Grader reward for this step (partial credit, strictly between 0 and 1).",
    )
    done: bool = Field(
        default=False,
        description="True when the incident has been resolved and episode ends.",
    )
    active_alerts: List[str] = Field(
        default_factory=list,
        description="List of currently active PagerDuty-style alerts firing in the cluster.",
    )
    task_description: str = Field(
        default="",
        description="Natural-language description of the current incident objective.",
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
        default=0.0,
        description="Running grader score based on correct diagnostic steps taken so far.",
    )
    live_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Real-time system telemetry snapshot: cpu, memory, error_rate.",
    )
    services_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Health status of each microservice: 'healthy', 'degraded', or 'critical'.",
    )
