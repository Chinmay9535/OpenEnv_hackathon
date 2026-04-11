"""
Grader for Task 1: Cache Memory Spike — OOM in compaction (Easy).

The agent must:
  1. query_metrics(service=cache, metric=memory_usage)
  2. fetch_logs(service=cache)
  3. resolve_incident(resolution_notes=<oom/heap related>)

Scoring rubric (cumulative, strictly in (0.001, 0.981)):
  - Step 1 correct: +0.28
  - Step 2 correct: +0.30
  - resolve_incident called: +0.33
  - Maximum achievable: 0.001 + 0.28 + 0.30 + 0.33 = 0.911
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
    Grade a Task 1 episode trajectory.

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
