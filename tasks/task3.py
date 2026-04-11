"""
Grader for Task 3: Frontend Timeouts — DB Deadlock (Hard).

The agent must trace a cascading failure:
  1. fetch_logs(service=frontend)
  2. fetch_logs(service=cart-service)
  3. run_db_query(SELECT * FROM pg_stat_activity WHERE state='active')
  4. run_db_query(SELECT pg_terminate_backend(9942))
  5. resolve_incident(resolution_notes=<deadlock/PID related>)

Scoring rubric (cumulative, strictly in (0.001, 0.981)):
  - Step 1 correct: +0.16
  - Step 2 correct: +0.16
  - Step 3 correct: +0.18
  - Step 4 correct: +0.22
  - resolve_incident called: +0.20
  - Maximum achievable: 0.001 + 0.16 + 0.16 + 0.18 + 0.22 + 0.20 = 0.921
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
    Grade a Task 3 episode trajectory.

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
