"""
Cloud SRE Simulator — Core Environment Logic.

Simulates a microservices cluster experiencing real-world infrastructure
incidents. Implements a rubric-based grader that assigns partial rewards
at each diagnostic step, providing dense RL training signal.

Grader Design Philosophy:
    Rewards are cumulative and awarded at the moment of correct action,
    not solely at the final resolve_incident() call. This allows RL agents
    to learn causal relationships between diagnostic steps and outcomes.

    All episode scores are strictly between 0.001 and 0.981 to satisfy
    OpenEnv's score range validation (exclusive of 0 and 1).
"""

import json
import math
import time
from typing import Dict, Optional


class CloudSimulator:
    """
    Simulates a microservices cluster for SRE incident response training.

    Three incident scenarios with increasing complexity:
        Level 1 — Cache OOM (Easy):     3-step diagnosis chain
        Level 2 — Bad Deployment (Med): 3-step deployment rollback
        Level 3 — DB Deadlock (Hard):   5-step cascading failure trace

    Reward Structure (strictly in (0, 1) range):
        - Partial rewards assigned at each correct diagnostic step
        - Wrong-service actions return informative-but-unhelpful data (reward=0)
        - Score is cumulative; resolve_incident() adds final completion bonus
        - Maximum achievable score: 0.981 (never reaches 1.0)
    """

    # Score boundaries
    BASE_SCORE = 0.001      # Minimum non-zero score (avoids exactly 0.0)
    MAX_SCORE = 0.981       # Maximum score (avoids exactly 1.0)

    def __init__(self, task_level: int = 1):
        self.task_level = max(1, min(3, task_level))
        self.step_count = 0
        self.created_at = time.time()
        self.resolved = False
        self.output = "Environment initialized."
        self.score = self.BASE_SCORE

        # Granular grader state — tracks which rubric items have been satisfied
        self.grader_state: Dict[str, bool] = {
            # Task 1 rubric
            "t1_queried_cache_memory": False,
            "t1_fetched_cache_logs": False,
            "t1_resolved": False,
            # Task 2 rubric
            "t2_listed_deployments": False,
            "t2_rolled_back_correctly": False,
            "t2_resolved": False,
            # Task 3 rubric
            "t3_traced_frontend": False,
            "t3_traced_cart": False,
            "t3_queried_db_activity": False,
            "t3_killed_deadlock": False,
            "t3_resolved": False,
        }

        self._setup_scenario()

    def _setup_scenario(self):
        """Configure the incident scenario for the current task level."""
        if self.task_level == 1:
            self.alerts = ["CRITICAL: High Memory Usage on `cache` service (95% heap utilization)"]
            self.task_desc = (
                "Task 1 — Alert Triage [Easy]: The `cache` service is experiencing a memory spike. "
                "Investigate using metrics and log data to identify the root cause, then resolve the incident."
            )
            self._services_status = {
                "cache": "critical",
                "payment-gateway": "healthy",
                "cart-service": "healthy",
                "frontend": "healthy",
                "database": "healthy",
            }

        elif self.task_level == 2:
            self.alerts = ["CRITICAL: High 500 Error Rate on `payment-gateway` (>40% requests failing)"]
            self.task_desc = (
                "Task 2 — Bad Deployment Rollback [Medium]: The `payment-gateway` service is returning "
                "HTTP 500 errors following a recent deployment. Check deployment history, identify the "
                "faulty version, and rollback to the last stable release."
            )
            self._services_status = {
                "cache": "healthy",
                "payment-gateway": "critical",
                "cart-service": "healthy",
                "frontend": "degraded",
                "database": "healthy",
            }

        elif self.task_level == 3:
            self.alerts = [
                "CRITICAL: Frontend Timeout Spikes (p99 latency > 30s)",
                "WARNING: cart-service response times elevated",
            ]
            self.task_desc = (
                "Task 3 — Cascading Database Failure [Hard]: Users are experiencing severe frontend timeouts. "
                "The failure appears to cascade from an upstream service. Trace the dependency chain through "
                "logs, identify the blocked database transaction, and terminate the deadlock."
            )
            self._services_status = {
                "cache": "healthy",
                "payment-gateway": "healthy",
                "cart-service": "critical",
                "frontend": "critical",
                "database": "degraded",
            }

    @property
    def services_status(self) -> Dict[str, str]:
        """Current health status of all microservices."""
        return self._services_status.copy()

    @property
    def live_metrics(self) -> Dict[str, float]:
        """Real-time system telemetry with realistic noise simulation."""
        t = time.time()
        noise = math.sin(t * 0.5) * 4 + math.cos(t * 1.3) * 2
        elapsed = min(120, t - self.created_at)

        cpu = 28 + noise
        mem = 38 + noise * 0.5
        err = max(0.1, 0.8 + noise * 0.15)

        if not self.resolved:
            if self.task_level == 1:
                mem = min(99.0, 68 + (elapsed * 0.18) + noise)
                cpu = min(75.0, 45 + (elapsed * 0.1) + noise)
            elif self.task_level == 2:
                err = min(78.0, 22 + (elapsed * 0.45) + noise)
                cpu = min(80.0, 50 + (elapsed * 0.12) + noise)
            elif self.task_level == 3:
                cpu = min(99.0, 80 + (elapsed * 0.12) + noise)
                err = min(55.0, 12 + (elapsed * 0.28) + noise)
                mem = min(85.0, 55 + (elapsed * 0.1) + noise)

        return {
            "cpu": round(max(0, cpu), 1),
            "memory": round(max(0, mem), 1),
            "error_rate": round(max(0, err), 1),
        }

    def _add_score(self, points: float) -> None:
        """Add points to score, capping at MAX_SCORE."""
        self.score = min(self.MAX_SCORE, self.score + points)

    # -------------------------------------------------------------------------
    # Action Handlers
    # -------------------------------------------------------------------------

    def query_metrics(self, service: Optional[str], metric: Optional[str]) -> str:
        """Query time-series metrics for a specific service."""
        self.step_count += 1
        service = (service or "").lower()
        metric = (metric or "").lower()

        points = []
        now = time.time()
        for i in range(10):
            t_offset = (10 - i) * 10
            t = now - t_offset
            noise = math.sin(t * 0.5) * 4

            # Task 1: Cache memory_usage
            if self.task_level == 1 and "cache" in service and "memory" in metric:
                if not self.grader_state["t1_queried_cache_memory"]:
                    self.grader_state["t1_queried_cache_memory"] = True
                    self._add_score(0.28)  # Partial credit for correct investigation step
                val = min(99, 55 + (i * 4) + noise) if not self.resolved else 38 + noise

            # Task 2: payment-gateway error_rate
            elif self.task_level == 2 and "payment" in service and "error" in metric:
                if not self.grader_state.get("t2_queried_metrics"):
                    self.grader_state["t2_queried_metrics"] = True
                    self._add_score(0.10)  # Minor credit — listing deployments is the key step
                val = min(75, 18 + (i * 6) + noise) if not self.resolved else 0.5 + noise

            else:
                # Wrong service or metric — return normal-looking data (informative negative)
                val = 28 + noise

            val = round(max(0, min(100, val)), 1)
            points.append({"time": f"T-{t_offset}s", "val": f"{val}%"})

        return json.dumps(points, indent=2)

    def fetch_logs(self, service: Optional[str], lines: Optional[int]) -> str:
        """Tail recent log lines from a service."""
        self.step_count += 1
        service = (service or "").lower()
        lines = lines or 20
        curr_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # Task 1: Cache logs reveal OOM in compaction
        if self.task_level == 1 and "cache" in service:
            if not self.grader_state["t1_fetched_cache_logs"]:
                self.grader_state["t1_fetched_cache_logs"] = True
                self._add_score(0.30)
            return (
                f"[{curr_time}] INFO  cache-service: Starting scheduled cache compaction (generation=3)\n"
                f"[{curr_time}] INFO  cache-service: Compaction progress: 24% ...\n"
                f"[{curr_time}] ERROR cache-service: java.lang.OutOfMemoryError: Java heap space\n"
                f"[{curr_time}] ERROR cache-service: \tat com.cache.Compactor.compact(Compactor.java:847)\n"
                f"[{curr_time}] FATAL cache-service: Compaction aborted. Memory pressure critical.\n"
                f"[{curr_time}] WARN  cache-service: Heap utilization at 95%. GC overhead limit exceeded.\n"
            )

        # Task 3: Frontend logs reveal upstream timeout to cart-service
        elif self.task_level == 3 and "frontend" in service:
            if not self.grader_state["t3_traced_frontend"]:
                self.grader_state["t3_traced_frontend"] = True
                self._add_score(0.16)
            return (
                f"[{curr_time}] INFO  frontend: GET /cart rendered in 78ms\n"
                f"[{curr_time}] WARN  frontend: Upstream cart-service response time > 5s\n"
                f"[{curr_time}] ERROR frontend: Timeout upstream waiting for cart-service (timeout=30s exceeded)\n"
                f"[{curr_time}] ERROR frontend: HTTP 504 Gateway Timeout returned to client\n"
                f"[{curr_time}] ERROR frontend: Upstream cart-service unresponsive. Circuit breaker OPEN.\n"
            )

        # Task 3: Cart-service logs reveal DB connection stall
        elif self.task_level == 3 and "cart" in service:
            if not self.grader_state["t3_traced_cart"]:
                self.grader_state["t3_traced_cart"] = True
                self._add_score(0.16)
            return (
                f"[{curr_time}] INFO  cart-service: Processing GET /api/cart/user/4821\n"
                f"[{curr_time}] WARN  cart-service: DB connection pool exhausted (pool_size=20, waiting=17)\n"
                f"[{curr_time}] ERROR cart-service: DB connection timeout after 30000ms fetching cart items\n"
                f"[{curr_time}] ERROR cart-service: Transaction stalled — lock wait timeout exceeded\n"
                f"[{curr_time}] FATAL cart-service: Deadlock detected. Rolling back transaction 0x7f3a.\n"
            )

        # Wrong service — return nominal logs (informative: this service is fine)
        return (
            f"[{curr_time}] INFO  {service}: All operations nominal.\n"
            f"[{curr_time}] INFO  {service}: Health check passed. Response time 12ms.\n"
            f"Showing last {lines} lines — no anomalies detected.\n"
        )

    def list_deployments(self, service: Optional[str]) -> str:
        """List recent deployments for a service."""
        self.step_count += 1
        service = (service or "").lower()

        if self.task_level == 2 and "payment" in service:
            if not self.grader_state["t2_listed_deployments"]:
                self.grader_state["t2_listed_deployments"] = True
                self._add_score(0.22)
            return (
                "Deployment history for payment-gateway:\n"
                "  v1.0.4  — deployed 47 mins ago   [CURRENT]  ← error rate spike started ~45 mins ago\n"
                "  v1.0.3  — deployed 3 days ago    [STABLE]   ← last known good version\n"
                "  v1.0.2  — deployed 12 days ago   [ARCHIVED]\n"
                "\nRecommendation: v1.0.4 deployment correlates with error onset. Consider rollback to v1.0.3."
            )

        return f"Deployment history for {service}:\n  v1.0.0  — deployed 14 days ago  [STABLE]\n"

    def rollback_deployment(self, service: Optional[str], version: Optional[str]) -> str:
        """Rollback a service to a previous deployment version."""
        self.step_count += 1
        service = (service or "").lower()
        version = (version or "").lower()

        if self.task_level == 2 and "payment" in service and "1.0.3" in version:
            if not self.grader_state["t2_rolled_back_correctly"]:
                self.grader_state["t2_rolled_back_correctly"] = True
                self._add_score(0.35)
            self.alerts = []
            self._services_status["payment-gateway"] = "healthy"
            self._services_status["frontend"] = "healthy"
            return (
                f"✓ Rollback initiated: payment-gateway v1.0.4 → v1.0.3\n"
                f"✓ New pods coming online... (0/3 ready)\n"
                f"✓ Health checks passing. Error rate dropping: 42% → 18% → 3% → 0.4%\n"
                f"✓ Rollback complete. payment-gateway v1.0.3 is now LIVE.\n"
                f"✓ Alert cleared: 500 error rate back to baseline.\n"
            )

        if "payment" in service and version:
            return (
                f"✗ Rollback to {version} failed. Version not found in stable registry.\n"
                f"  Available stable versions: v1.0.3, v1.0.2\n"
            )

        return f"✗ Rollback failed: service '{service}' not found or insufficient permissions.\n"

    def run_db_query(self, query: Optional[str]) -> str:
        """Execute a database management query."""
        self.step_count += 1
        q = (query or "").lower().strip()

        if self.task_level == 3:
            # Correct: inspect active transactions
            if "pg_stat_activity" in q and ("select" in q or "select" in q):
                if not self.grader_state["t3_queried_db_activity"]:
                    self.grader_state["t3_queried_db_activity"] = True
                    self._add_score(0.18)
                return (
                    " pid  | state  | wait_event_type | wait_event | duration | query\n"
                    "------+--------+-----------------+------------+----------+-------------------------------\n"
                    " 9942 | active | Lock            | relation   | 00:08:43 | UPDATE carts SET is_checked_out=TRUE WHERE ...\n"
                    " 9938 | idle   | Client          | ClientRead | 00:00:01 | SELECT * FROM sessions WHERE ...\n"
                    " 9945 | active | Lock            | tuple      | 00:08:41 | SELECT id FROM carts WHERE user_id=4821 ...\n"
                    "\n⚠ PID 9942 has been holding a row-level lock for 8m43s — likely cause of deadlock cascade.\n"
                )

            # Correct: terminate the deadlocked transaction
            if ("terminate_backend" in q or "kill" in q) and "9942" in q:
                if not self.grader_state["t3_killed_deadlock"]:
                    self.grader_state["t3_killed_deadlock"] = True
                    self._add_score(0.22)
                self.alerts = []
                self._services_status["database"] = "healthy"
                self._services_status["cart-service"] = "healthy"
                self._services_status["frontend"] = "healthy"
                return (
                    "pg_terminate_backend\n"
                    "---------------------\n"
                    " t\n"
                    "(1 row)\n\n"
                    "✓ Transaction PID 9942 terminated successfully.\n"
                    "✓ Lock released. Waiting transactions (9945, 9943, 9941) proceeding.\n"
                    "✓ cart-service connection pool draining backlog...\n"
                    "✓ frontend upstream latency recovering: 28s → 4s → 0.3s\n"
                )

        return "Query executed. 0 rows returned.\n"

    def resolve_incident(self, notes: Optional[str]) -> str:
        """Submit an incident resolution with diagnosis notes."""
        self.step_count += 1
        notes = notes or "No notes provided."

        # Determine completion bonus based on how thorough the diagnosis was
        if self.task_level == 1:
            both_steps_done = (
                self.grader_state["t1_queried_cache_memory"]
                and self.grader_state["t1_fetched_cache_logs"]
            )
            if both_steps_done and not self.grader_state["t1_resolved"]:
                self.grader_state["t1_resolved"] = True
                self._add_score(0.33)  # Full completion bonus → total ~0.92
            elif not self.grader_state["t1_resolved"]:
                # Partial: resolved without full diagnosis
                self.grader_state["t1_resolved"] = True
                self._add_score(0.05)

        elif self.task_level == 2:
            rolled_back = self.grader_state["t2_rolled_back_correctly"]
            if rolled_back and not self.grader_state["t2_resolved"]:
                self.grader_state["t2_resolved"] = True
                self._add_score(0.30)  # Full completion bonus → total ~0.88
            elif not self.grader_state["t2_resolved"]:
                self.grader_state["t2_resolved"] = True
                self._add_score(0.05)

        elif self.task_level == 3:
            deadlock_killed = self.grader_state["t3_killed_deadlock"]
            full_trace = (
                self.grader_state["t3_traced_frontend"]
                and self.grader_state["t3_traced_cart"]
                and self.grader_state["t3_queried_db_activity"]
            )
            if deadlock_killed and full_trace and not self.grader_state["t3_resolved"]:
                self.grader_state["t3_resolved"] = True
                self._add_score(0.20)  # Full completion → total ~0.93
            elif deadlock_killed and not self.grader_state["t3_resolved"]:
                self.grader_state["t3_resolved"] = True
                self._add_score(0.12)
            elif not self.grader_state["t3_resolved"]:
                self.grader_state["t3_resolved"] = True
                self._add_score(0.04)

        self.resolved = True
        self.alerts = []
        self._services_status = {s: "healthy" for s in self._services_status}

        return (
            f"✓ Incident resolved.\n"
            f"  Resolution notes: {notes}\n"
            f"  Diagnostic score: {self.score:.3f}\n"
            f"  All alerts cleared. System stabilized.\n"
        )

    def step(self, action) -> str:
        """Route an action to the appropriate handler."""
        if self.resolved:
            return "Incident already resolved. Reset the environment to start a new episode."

        action_type = getattr(action, "action_type", None)

        if action_type == "query_metrics":
            return self.query_metrics(action.service, action.metric)
        elif action_type == "fetch_logs":
            return self.fetch_logs(action.service, action.lines)
        elif action_type == "list_deployments":
            return self.list_deployments(action.service)
        elif action_type == "rollback_deployment":
            return self.rollback_deployment(action.service, action.version)
        elif action_type == "run_db_query":
            return self.run_db_query(action.query)
        elif action_type == "resolve_incident":
            return self.resolve_incident(action.resolution_notes)
        elif action_type == "noop":
            self.step_count += 1
            return "No operation performed."
        else:
            self.step_count += 1
            return f"Unknown action type: '{action_type}'. Valid types: query_metrics, fetch_logs, list_deployments, rollback_deployment, run_db_query, resolve_incident."
