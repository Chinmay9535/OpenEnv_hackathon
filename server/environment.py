"""
Cloud SRE Environment Implementation.

Wraps the CloudSimulator into an openenv-compliant Environment interface.
Each episode presents one SRE incident scenario. The agent must diagnose
the root cause using the available tools and resolve the incident.

Episode lifecycle:
    reset(task_id) → initial observation with active alert
    step(action)   → observation + incremental reward from rubric grader
    done=True      → episode ends after resolve_incident() is called
"""

from typing import Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    # Fallback for local dev without full openenv install
    class Environment:  # type: ignore
        pass

    class State:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

try:
    from .models import CloudSREAction, CloudSREObservation
    from .simulator import CloudSimulator
except ImportError:
    from models import CloudSREAction, CloudSREObservation  # type: ignore
    from simulator import CloudSimulator  # type: ignore


class CloudSREEnvironment(Environment):
    """
    Cloud SRE environment for training agents on infrastructure incident response.

    The environment simulates a realistic microservices cluster experiencing
    progressively complex incidents. An AI agent acts as an on-call SRE,
    using diagnostic tools to find and fix the root cause.

    Three task levels:
        task_id=1 (Easy)   — Cache memory spike → OOM in compaction
        task_id=2 (Medium) — Payment-gateway 500s → Bad deployment rollback
        task_id=3 (Hard)   — Frontend timeouts → Cart → DB deadlock (PID kill)

    Reward structure:
        Rewards are assigned at each step proportional to correct diagnostic
        actions, not just at the final resolve_incident call. This provides
        dense RL training signal throughout the episode.

    Example:
        >>> env = CloudSREEnvironment()
        >>> obs = env.reset(task_id=1)
        >>> print(obs.active_alerts)   # ['CRITICAL: High Memory Usage on `cache` service']
        >>> action = CloudSREAction(action_type="query_metrics", service="cache", metric="memory_usage")
        >>> obs = env.step(action)
        >>> print(obs.reward)          # 0.30  (partial credit for correct diagnostic step)
        >>> print(obs.done)            # False
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize without starting an episode. Call reset() to begin."""
        self._sim: Optional[CloudSimulator] = None
        self._episode_id: str = str(uuid4())
        self._step_count: int = 0

    def reset(
        self,
        task_id: int = 1,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CloudSREObservation:
        """Reset the environment and start a new incident episode.

        Args:
            task_id: Incident difficulty level (1=Easy, 2=Medium, 3=Hard).
            episode_id: Optional episode identifier for tracking.

        Returns:
            Initial CloudSREObservation with the incident alert and task description.
        """
        # Clamp task_id to valid range
        task_id = max(1, min(3, int(task_id)))

        self._sim = CloudSimulator(task_level=task_id)
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0

        obs = CloudSREObservation(
            active_alerts=self._sim.alerts,
            task_description=self._sim.task_desc,
            last_action_output="Environment Reset. New incident detected. Begin diagnosis.",
            step_count=0,
            cumulative_score=0.001,
            live_metrics=self._sim.live_metrics,
            services_status=self._sim.services_status,
            reward=0.001,  # BASE_SCORE — never exactly 0.0
            done=False,
        )
        return obs

    def step(self, action: CloudSREAction) -> CloudSREObservation:
        """Execute one SRE action and return the resulting observation.

        Args:
            action: A CloudSREAction describing the command to execute.

        Returns:
            CloudSREObservation with command output, current alerts, and reward.
        """
        if self._sim is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1
        output = self._sim.step(action)

        obs = CloudSREObservation(
            active_alerts=self._sim.alerts,
            task_description=self._sim.task_desc,
            last_action_output=output[:2000],  # truncate very long outputs
            step_count=self._step_count,
            cumulative_score=self._sim.score,
            live_metrics=self._sim.live_metrics,
            services_status=self._sim.services_status,
            reward=self._sim.score,
            done=self._sim.resolved,
        )
        return obs

    @property
    def state(self) -> State:
        """Current episode state for openenv framework tracking."""
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )
