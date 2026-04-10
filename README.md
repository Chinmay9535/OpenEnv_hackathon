---
title: Cloud SRE War Room
emoji: 🚨
colorFrom: red
colorTo: indigo
sdk: docker
pinned: true
---

# 🚨 Cloud SRE War Room — OpenEnv Environment

> **An AI-native SRE incident response environment for RL training.** Train agents to diagnose and resolve real-world infrastructure incidents across a simulated microservices cluster — the same challenges faced daily by Site Reliability Engineers at companies like Google, Netflix, and Meta.

---

## 🎯 Real-World Utility

Unlike toy environments, SRE incident response is a **genuine, high-value domain** for AI agents:

- Companies spend **$300K+ per hour** on unresolved P1 incidents
- SRE engineers follow structured diagnostic workflows (the [Google SRE Book](https://sre.google/sre-book/table-of-contents/) documents these)
- An AI agent trained here can generalize across Datadog, PagerDuty, Grafana, and K8s tooling

This environment faithfully models the SRE workflow: **receive alert → query metrics → inspect logs → correlate dependencies → remediate root cause**.

---

## 🖥 War Room Dashboard

The environment ships with a premium **live monitoring dashboard** accessible at the root URL (`/`):

- Real-time CPU, Memory, and Error Rate charts (Chart.js, 2s refresh)
- Active alert panel with severity coloring
- Microservice health status grid
- Agent action log stream

> **Access the War Room:** Navigate to the Space URL or run locally at `http://localhost:7860`

---

## ⚙ Environment Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cloud SRE OpenEnv                            │
│                                                                 │
│  inference.py ──── POST /reset ────► CloudSREEnvironment        │
│       │                                      │                  │
│       │            POST /step ─────► CloudSimulator             │
│       │                                      │                  │
│       └────── [START][STEP][END] ◄─── reward + observation      │
│                                                                 │
│  War Room Dashboard (/) ◄──── live_metrics + services_status    │
└─────────────────────────────────────────────────────────────────┘
```

### Action Space (`CloudSREAction`)

```json
{
  "action_type": "query_metrics | fetch_logs | list_deployments | rollback_deployment | run_db_query | resolve_incident",
  "service":     "cache | payment-gateway | cart-service | frontend | database",
  "metric":      "memory_usage | error_rate | cpu_usage | latency_p99",
  "lines":       20,
  "version":     "v1.0.3",
  "query":       "SELECT * FROM pg_stat_activity WHERE state='active'",
  "resolution_notes": "Root cause identified: OOM in cache compaction..."
}
```

### Observation Space (`CloudSREObservation`)

```json
{
  "active_alerts":       ["CRITICAL: High Memory Usage on `cache` service"],
  "task_description":    "Task 1 — Alert Triage: The cache service...",
  "last_action_output":  "T-100s: 51.7%\nT-90s: 60.2%...",
  "step_count":          2,
  "cumulative_score":    0.581,
  "live_metrics":        {"cpu": 44.1, "memory": 78.3, "error_rate": 0.9},
  "services_status":     {"cache": "critical", "payment-gateway": "healthy"}
}
```

---

## 📋 Task Progression

### Task 1 — Alert Triage `[Easy]`
**Incident:** `cache` service memory at 95% and climbing  
**Root Cause:** Java heap exhausted during scheduled cache compaction (`OutOfMemoryError`)  
**Optimal Path:** `query_metrics(cache, memory_usage)` → `fetch_logs(cache)` → `resolve_incident`  
**Grader Rubric:** +0.28 for correct metric query, +0.30 for log fetch, +0.33 for full resolution  
**Baseline Score:** 0.911

### Task 2 — Bad Deployment Rollback `[Medium]`
**Incident:** `payment-gateway` 500 error rate >40% after recent deployment  
**Root Cause:** Version v1.0.4 introduced a regression; v1.0.3 is the last stable build  
**Optimal Path:** `list_deployments(payment-gateway)` → `rollback_deployment(payment-gateway, v1.0.3)` → `resolve_incident`  
**Grader Rubric:** +0.22 for listing deployments, +0.35 for correct rollback, +0.30 for resolution  
**Baseline Score:** 0.871

### Task 3 — Cascading Database Failure `[Hard]`
**Incident:** Frontend p99 latency >30s; cascades through cart-service to a deadlocked DB  
**Root Cause:** Transaction PID 9942 holding a row lock for 8m43s, blocking the cart checkout flow  
**Optimal Path:** `fetch_logs(frontend)` → `fetch_logs(cart-service)` → `run_db_query(SELECT pg_stat_activity)` → `run_db_query(pg_terminate_backend(9942))` → `resolve_incident`  
**Grader Rubric:** +0.16 per trace step, +0.18 for DB activity query, +0.22 for deadlock kill, +0.20 for resolution  
**Baseline Score:** 0.921

---

## 🏆 Reward Design

Rewards are **assigned at each diagnostic step** (not just at episode end), providing dense RL signal:

```
Step 1: query correct metrics    → reward = 0.281  (incremental progress)
Step 2: fetch correct logs       → reward = 0.581  (further progress)  
Step 3: resolve with full trace  → reward = 0.911  (episode complete)
```

- All rewards strictly in **(0, 1)** — never exactly 0.0 or 1.0
- Wrong-service actions return realistic but unhelpful data (zero reward, non-zero information)
- Partial credit if agent skips steps but eventually resolves the incident

---

## 🚀 Running Locally

```bash
# Clone and install
git clone https://github.com/Chinmay9535/OpenEnv_hackathon
cd OpenEnv_hackathon
uv sync

# Terminal 1 — Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Terminal 2 — Run the baseline agent
python inference.py
```

**Environment variables (injected automatically during evaluation):**
```
API_BASE_URL  — LiteLLM proxy endpoint (default: http://localhost:8000/v1)
MODEL_NAME    — Model for inference (default: gpt-4o)
HF_TOKEN      — Required HuggingFace token
API_KEY       — Meta evaluation proxy key (injected at runtime)
```

**Docker:**
```bash
docker build -t cloud-sre-env .
docker run -p 7860:7860 -e HF_TOKEN=<your_token> cloud-sre-env
```

---

## 📁 Project Structure

```
OpenEnv_hackathon/
├── inference.py           # Baseline SRE agent (spec-compliant output format)
├── openenv.yaml           # OpenEnv environment specification
├── Dockerfile             # Container definition
├── pyproject.toml         # Python dependencies
├── server/
│   ├── app.py             # FastAPI server (reset/step/state/schema endpoints)
│   ├── environment.py     # CloudSREEnvironment — openenv.core.Environment subclass
│   ├── simulator.py       # CloudSimulator — incident logic + rubric grader
│   ├── models.py          # CloudSREAction + CloudSREObservation (openenv types)
│   └── index.html         # War Room glassmorphism dashboard
└── README.md
```

---

## 📊 OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| `POST /reset` endpoint | ✅ |
| `POST /step` endpoint | ✅ |
| `GET /state` endpoint | ✅ |
| `GET /schema` endpoint | ✅ |
| `Action` inherits `openenv.core.env_server.types.Action` | ✅ |
| `Observation` inherits `openenv.core.env_server.types.Observation` | ✅ |
| `Environment` inherits `openenv.core.env_server.interfaces.Environment` | ✅ |
| Reward strictly in (0, 1) | ✅ |
| 3+ tasks with graders | ✅ |
| `inference.py` output format compliant | ✅ |
| Docker build | ✅ |
