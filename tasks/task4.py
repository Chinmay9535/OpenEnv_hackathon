"""
Grader for Task 4: API Gateway Rate-Limit Cascade (Hard).

The agent must:
  1. query_metrics(service=auth-service, metric=cpu_usage)
  2. fetch_logs(service=auth-service)
  3. rollback_deployment(service=auth-service, version=v2.3.0)
  4. resolve_incident(resolution_notes=<auth/rollback/cpu related>)

Scoring rubric (cumulative, strictly in (0.001, 0.981)):
  - Step 1 correct: +0.20
  - Step 2 correct: +0.22
  - Step 3 correct: +0.28
  - resolve_incident called: +0.20
  - Maximum achievable: 0.001 + 0.20 + 0.22 + 0.28 + 0.20 = 0.901
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
    Grade a Task 4 episode trajectory.

    Args:
        trajectory: List of (action_dict, observation_dict) tuples recorded
                    during the episode.

    Returns:
        Final task score, strictly in (0, 1).
    """
    if not trajectory:
        return _MIN_SCORE

    # Extract final reward from the last observation
    _action, final_obs = trajectory[-1]
    reward = final_obs.get("reward", _MIN_SCORE)

    try:
        score = float(reward)
    except (TypeError, ValueError):
        score = _MIN_SCORE

    return _clamp(score)
