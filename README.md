# Cloud SRE / DevOps OpenEnv Environment

An advanced, real-world Site Reliability Engineering environment built for the Meta OpenEnv Hackathon. This environment evaluates autonomous AI agents on their ability to triage, diagnose, and resolve production cloud infrastructure incidents.

## Environment Description 
Unlike simple customer support bots or toy data tasks, managing infrastructure is a challenging, high-value domain for AI. The `cloud-sre-env` simulates a microservices architecture. The agent acts as an on-call SRE and must interact with the system to recover from cascading failures. 

The environment tests the agent on 3 progressive tasks:
- **Task 1 (Easy):** Alert Triage. The `cache` service experiences a memory spike. The agent must fetch the correct metrics, retrieve logs, identify the OOM error, and resolve the incident.
- **Task 2 (Medium):** Bad Deployment Rollback. The `payment-gateway` starts returning HTTP 500 errors. The agent detects a recent deployment, locates the previous stable version, and executes a rollback action.
- **Task 3 (Hard):** Cascading Database Failure. Frontend timeouts lead back to a cart service failure caused by a deadlocked database transaction. The agent must trace the error through the microservices, identify the blocking query, and execute a `KILL` operation on the database.

## Action & Observation Spaces

### Action Space (`CloudSREAction`)
The agent communicates via JSON objects matching this Pydantic schema:
- `action_type`: One of `query_metrics`, `fetch_logs`, `list_deployments`, `rollback_deployment`, `run_db_query`, `resolve_incident`.
- `service`: Target microservice (e.g., `payment-gateway`, `frontend`, `cart-service`, `cache`).
- `metric`: Metric to visualize (e.g., `error_rate`, `memory_usage`).
- `lines`: Lines of logs to tail.
- `version`: Version to rollback to.
- `query`: SQL query to run.
- `resolution_notes`: Explanation sent when resolving the incident.

### Observation Space (`CloudSREObservation`)
- `active_alerts`: A list of currently firing system alerts.
- `task_description`: The current debug task description.
- `last_action_output`: The text output (JSON metrics, log lines, or command results) from the previous action.
- `echoed_message`: System status echo.

## Setup Instructions

**Prerequisites:**
- Python 3.12+
- `openenv-core`
- Docker

**Running Locally:**
1. Clone the repository.
2. Run `uvicorn server.app:app --host 0.0.0.0 --port 8000` to start the HTTP environment simulator.
3. Run `python inference.py` in a separate terminal to run the baseline agent script.

**Docker Deployment:**
```bash
docker build -t cloud-sre-env .
docker run -p 8000:8000 cloud-sre-env
```

## Baseline Scores
The baseline script `inference.py` deterministically solves the easy task to establish a working environment connection.
- **Task 1:** 1.0 (100%) - Successfully checks metrics, reads logs, and resolves the issue.
- **Task 2:** Tested internally - Requires active OpenAI agent logic capable of chaining the `list_deployments` -> `rollback_deployment` commands.
- **Task 3:** Tested internally - Requires complex multi-hop diagnosis capabilities to query `pg_stat_activity`. 

Evaluators testing the environment manually can send `{"task_id": 3}` to the `/reset` endpoint to experience the cascading failure.
