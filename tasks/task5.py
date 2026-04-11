"""
Grader for Task 5: Kubernetes Node OOM / Pod Eviction Storm (Expert).

The agent must:
  1. query_metrics(service=worker-node-3/cluster, metric=memory)
  2. fetch_logs(service=order-service/node/eviction)
  3. run_db_query(pg_stat_activity/connections)
  4. rollback_deployment(service=order-service, version=v3.1.9)
  5. resolve_incident(resolution_notes=<oom/rollback/order-service related>)

Scoring rubric (cumulative, strictly in (0.001, 0.981)):
  - Step 1 correct: +0.16
  - Step 2 correct: +0.18
  - Step 3 correct: +0.14
  - Step 4 correct: +0.24
  - resolve_incident called: +0.18
  - Maximum achievable: 0.001 + 0.16 + 0.18 + 0.14 + 0.24 + 0.18 = 0.901
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
    Grade a Task 5 episode trajectory.

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
