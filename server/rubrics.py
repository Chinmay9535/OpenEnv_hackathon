"""
SRE Incident Rubrics for reward computation.

Uses ExponentialDiscountingTrajectoryRubric so that RL training
infrastructure can extract per-step temporally-discounted rewards
(like the chess_env reference implementation), while still providing
dense per-step rewards for fine-grained agent feedback.
"""

from typing import Any, List, Optional, Tuple

try:
    from openenv.core.rubrics.trajectory import ExponentialDiscountingTrajectoryRubric
except (ImportError, ModuleNotFoundError):
    # Compatibility shim when installed core lacks rubrics
    class ExponentialDiscountingTrajectoryRubric:
        def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
            self.gamma = gamma
            self.intermediate_reward = intermediate_reward
            self._trajectory: List[Tuple[Any, Any]] = []
            self.last_score: Optional[float] = None

        def __call__(self, action: Any, observation: Any) -> float:
            self._trajectory.append((action, observation))
            if getattr(observation, "done", False):
                score = self.score_trajectory(self._trajectory)
                self.last_score = score
                return score
            return self.intermediate_reward

        def reset(self) -> None:
            self._trajectory = []
            self.last_score = None

        def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
            raise NotImplementedError

        def compute_step_rewards(self) -> List[float]:
            """Compute temporally-discounted step rewards for RL training."""
            if not self._trajectory:
                return []
            final_score = self.score_trajectory(self._trajectory)
            total_steps = len(self._trajectory)
            return [
                self.gamma ** (total_steps - 1 - step_idx) * final_score
                for step_idx in range(total_steps)
            ]


_MIN_SCORE = 0.001
_MAX_SCORE = 0.981


class SREIncidentRubric(ExponentialDiscountingTrajectoryRubric):
    """
    Rubric for scoring SRE incident-response trajectories.

    Reads the cumulative reward from the final observation (set by the
    CloudSimulator's built-in rubric grader) and returns it as the
    episode score, with exponential temporal discounting for RL training.

    Score range: strictly in (0.001, 0.981) to satisfy the validator.

    Usage:
        rubric = SREIncidentRubric(gamma=0.99)
        rubric.reset()
        for action, obs in episode:
            reward = rubric(action, obs)  # dense per-step signal
        step_rewards = rubric.compute_step_rewards()  # for RL PPO etc.
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """
        Read the cumulative grader score from the final observation.

        The CloudSimulator already maintains a point-based rubric grader
        internally; this method extracts its final value.

        Args:
            trajectory: List of (action, observation) tuples.

        Returns:
            Final task score, strictly in (0.001, 0.981).
        """
        if not trajectory:
            return _MIN_SCORE

        _, final_obs = trajectory[-1]

        # Try reward attribute first, fall back to dict lookup
        reward = getattr(final_obs, "reward", None)
        if reward is None and isinstance(final_obs, dict):
            reward = final_obs.get("reward", _MIN_SCORE)

        try:
            score = float(reward)
        except (TypeError, ValueError):
            score = _MIN_SCORE

        return max(_MIN_SCORE, min(_MAX_SCORE, score))
