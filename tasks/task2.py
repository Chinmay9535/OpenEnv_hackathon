"""
Grader for Task 2: Payment-Gateway 500 Errors — Bad Deployment Rollback (Medium).

The agent must:
  1. list_deployments(service=payment-gateway)
  2. rollback_deployment(service=payment-gateway, version=v1.0.3)
  3. resolve_incident(resolution_notes=<rollback related>)

Scoring rubric (cumulative, strictly in (0.001, 0.981)):
  - Step 1 correct: +0.22
  - Step 2 correct: +0.35
  - resolve_incident called: +0.30
  - Maximum achievable: 0.001 + 0.22 + 0.35 + 0.30 = 0.871
"""

from typing import Any, Dict, List, Tuple

# Score boundaries — must stay strictly between 0 and 1
_MIN_SCORE = 0.001
_MAX_SCORE = 0.981


def _clamp(score: float) -> float:
    """Guarantee score is strictly inside (0, 1)."""
    return max(_MIN_SCORE, min(_MAX_SCORE, float(score)))


def grade(trajectory: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> float:
    """
    Grade a Task 2 episode trajectory.

    Args:
        trajectory: List of (action_dict, observation_dict) tuples recorded
                    during the episode. The observation_dict must include a
                    'reward' key with the cumulative grader score.

    Returns:
        Final task score, strictly in (0, 1).
    """
    if not trajectory:
        return _MIN_SCORE

    # Extract final reward from the last observation
    _action, final_obs = trajectory[-1]
    reward = final_obs.get("reward", _MIN_SCORE)

    # Validate and clamp — ensures we never return exactly 0 or 1
    try:
        score = float(reward)
    except (TypeError, ValueError):
        score = _MIN_SCORE

    return _clamp(score)
