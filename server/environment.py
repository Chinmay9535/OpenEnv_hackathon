"""
Cloud SRE Environment Implementation.

Wraps CloudSimulator into an openenv-compliant Environment with proper
ExponentialDiscountingTrajectoryRubric support (matching the chess_env
reference implementation pattern from Meta's OpenEnv examples).

Five incident scenarios:
    task_id=1 — Cache OOM in compaction               [Easy]
    task_id=2 — Payment-gateway bad deployment rollback [Medium]
    task_id=3 — Frontend → Cart → DB deadlock          [Hard]
    task_id=4 — API Gateway rate-limit cascade (auth)  [Hard]
    task_id=5 — K8s node OOM + pod eviction storm      [Expert]
"""

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment as _OpenEnvEnvironment
    _BASE = _OpenEnvEnvironment
except ImportError:
    _BASE = object  # type: ignore

try:
    from .models import CloudSREAction, CloudSREObservation
    from .rubrics import SREIncidentRubric
    from .simulator import CloudSimulator, SCENARIOS
except ImportError:
    from models import CloudSREAction, CloudSREObservation  # type: ignore
    from rubrics import SREIncidentRubric                   # type: ignore
    from simulator import CloudSimulator, SCENARIOS         # type: ignore


class CloudSREEnvironment(_BASE):  # type: ignore
    """
    Cloud SRE environment for training agents on infrastructure incident response.

    Implements the full OpenEnv Environment interface with:
    - Dense per-step rewards from the CloudSimulator rubric grader
    - Exponential temporal discounting via SREIncidentRubric (RL training)
    - SUPPORTS_CONCURRENT_SESSIONS = True for parallel evaluations
    - Five progressively complex incident scenarios

    Example:
        >>> env = CloudSREEnvironment()
        >>> obs = env.reset(task_id=1)
        >>> print(obs.active_alerts)
        ['CRITICAL: High Memory Usage on `cache` service (95% heap utilization)']
        >>> action = CloudSREAction(action_type="query_metrics", service="cache", metric="memory_usage")
        >>> obs = env.step(action)
        >>> print(obs.reward)   # 0.281 (partial credit)
        >>> print(obs.done)     # False
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize without starting an episode. Call reset() to begin."""
        self._sim: Optional[CloudSimulator] = None
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0
        # Rubric for RL training infrastructure (temporal credit assignment)
        self.rubric = SREIncidentRubric(gamma=0.99)

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: int = 1,
        episode_id: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> CloudSREObservation:
        """
        Reset the environment and start a new incident episode.

        Args:
            task_id: Incident level (1=Easy … 5=Expert). Clamped to [1, 5].
            episode_id: Optional episode identifier for tracking.
            seed: Ignored (deterministic simulator for reproducibility).

        Returns:
            Initial CloudSREObservation with the incident alert and task description.
        """
        task_id = max(1, min(5, int(task_id)))

        self._sim          = CloudSimulator(task_level=task_id)
        self._episode_id   = episode_id or str(uuid4())
        self._step_count   = 0
        self.rubric.reset()  # Clear trajectory state for RL training

        obs = CloudSREObservation(
            active_alerts=self._sim.alerts,
            task_description=self._sim.task_desc,
            last_action_output=(
                "Environment reset. New incident detected. Begin triage.\n"
                f"Severity: {'CRITICAL' if task_id >= 3 else 'HIGH'} | "
                f"Task level: {task_id}/5"
            ),
            step_count=0,
            cumulative_score=0.001,
            live_metrics=self._sim.live_metrics,
            services_status=self._sim.services_status,
            reward=0.001,   # BASE_SCORE — strictly > 0.0
            done=False,
        )
        return obs

    def step(
        self,
        action: CloudSREAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CloudSREObservation:
        """
        Execute one SRE action and return the resulting observation.

        Args:
            action: A CloudSREAction with the tool to call and its parameters.
            timeout_s: Ignored (all actions are synchronous).

        Returns:
            CloudSREObservation with command output, updated metrics, and reward.
        """
        if self._sim is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._step_count += 1
        output = self._sim.step(action)

        obs = CloudSREObservation(
            active_alerts=self._sim.alerts,
            task_description=self._sim.task_desc,
            last_action_output=output[:3000],   # cap very long outputs
            step_count=self._step_count,
            cumulative_score=self._sim.score,
            live_metrics=self._sim.live_metrics,
            services_status=self._sim.services_status,
            reward=self._sim.score,             # cumulative rubric score
            done=self._sim.resolved,
        )

        # Feed into rubric for RL training trajectory tracking
        self._apply_rubric(action, obs)

        return obs

    @property
    def state(self) -> dict:  # type: ignore
        """Current episode state for openenv framework tracking."""
        task_level = self._sim.task_level if self._sim else 1
        return {
            "episode_id":  self._episode_id,
            "step_count":  self._step_count,
            "task_level":  task_level,
            "task_name":   SCENARIOS.get(task_level, {}).get("name", "unknown"),
            "resolved":    self._sim.resolved if self._sim else False,
        }

    def get_metadata(self):
        """Override metadata to provide rich environment description."""
        try:
            from openenv.core.env_server.types import EnvironmentMetadata
            return EnvironmentMetadata(
                name="cloud-sre-env",
                description=(
                    "Cloud SRE incident-response environment. An AI agent acts as an "
                    "on-call SRE diagnosing microservice failures across 5 progressively "
                    "complex scenarios: cache OOM, bad deployment rollback, DB deadlock, "
                    "auth-service CPU regression, and K8s node eviction storm."
                ),
                version="2.0.0",
            )
        except Exception:
            pass
