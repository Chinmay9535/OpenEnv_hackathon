"""
Cloud SRE Simulator — Core Environment Logic (v2 — Enhanced).

Simulates a microservices cluster experiencing real-world infrastructure
incidents across FIVE increasingly complex scenarios. Implements a
rubric-based grader with dense per-step rewards and realistic tool
outputs (metrics, logs, deployments, DB queries).

Grader Design:
    Rewards are cumulative and awarded at the moment of correct diagnostic
    action. Wrong-service actions return realistic but unhelpful data
    (informative-negative) to teach agents to read signals. All scores
    are strictly between 0.001 and 0.981 — never exactly 0 or 1.
"""

import json
import math
import random
import time
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS = {
    1: {
        "name": "cache-oom-triage",
        "title": "Cache Memory Spike — OOM in Compaction [Easy]",
        "alerts": [
            "CRITICAL: High Memory Usage on `cache` service (95% heap utilization)",
        ],
        "desc": (
            "Task 1 — Alert Triage [Easy]: The `cache` service is experiencing a memory "
            "spike. Investigate using metrics and log data to identify the OutOfMemoryError "
            "in compaction, then resolve the incident."
        ),
        "services": {
            "cache": "critical",
            "payment-gateway": "healthy",
            "cart-service": "healthy",
            "frontend": "healthy",
            "database": "healthy",
            "api-gateway": "healthy",
        },
    },
    2: {
        "name": "payment-gateway-rollback",
        "title": "Payment-Gateway 500 Errors — Bad Deployment [Medium]",
        "alerts": [
            "CRITICAL: High 500 Error Rate on `payment-gateway` (>40% requests failing)",
        ],
        "desc": (
            "Task 2 — Bad Deployment Rollback [Medium]: The `payment-gateway` service is "
            "returning HTTP 500 errors following a recent deployment. Check deployment "
            "history, identify the faulty version, and rollback to the last stable release."
        ),
        "services": {
            "cache": "healthy",
            "payment-gateway": "critical",
            "cart-service": "healthy",
            "frontend": "degraded",
            "database": "healthy",
            "api-gateway": "degraded",
        },
    },
    3: {
        "name": "frontend-db-deadlock",
        "title": "Frontend Timeouts — Cart-Service DB Deadlock [Hard]",
        "alerts": [
            "CRITICAL: Frontend Timeout Spikes (p99 latency > 30s)",
            "WARNING: cart-service response times elevated",
        ],
        "desc": (
            "Task 3 — Cascading Database Failure [Hard]: Users are experiencing severe "
            "frontend timeouts. The failure cascades from cart-service through a blocked "
            "PostgreSQL transaction. Trace the dependency chain through logs, identify the "
            "deadlocked PID, and terminate it."
        ),
        "services": {
            "cache": "healthy",
            "payment-gateway": "healthy",
            "cart-service": "critical",
            "frontend": "critical",
            "database": "degraded",
            "api-gateway": "healthy",
        },
    },
    4: {
        "name": "api-gateway-rate-limit-cascade",
        "title": "API Gateway Rate-Limit Cascade — Auth-Service CPU Spike [Hard]",
        "alerts": [
            "CRITICAL: API Gateway returning HTTP 429 Too Many Requests (>60% of traffic)",
            "CRITICAL: auth-service CPU utilization at 98%",
            "WARNING: Downstream service latency elevated across all endpoints",
        ],
        "desc": (
            "Task 4 — Rate Limit Cascade [Hard]: The API Gateway is throttling the majority "
            "of incoming traffic. Investigation suggests the auth-service is CPU-bound. "
            "Query auth-service metrics, inspect logs for the hot code path, rollback the "
            "auth-service deployment that introduced the regression, then resolve."
        ),
        "services": {
            "cache": "healthy",
            "payment-gateway": "degraded",
            "cart-service": "degraded",
            "frontend": "critical",
            "database": "healthy",
            "api-gateway": "critical",
            "auth-service": "critical",
        },
    },
    5: {
        "name": "k8s-node-oom-eviction",
        "title": "Kubernetes Node OOM — Pod Eviction Storm [Expert]",
        "alerts": [
            "CRITICAL: Kubernetes node `worker-node-3` OOMKilled — 8 pods evicted",
            "CRITICAL: order-service replicas reduced to 0 (all pods evicted)",
            "WARNING: Horizontal Pod Autoscaler failing (pending pods cannot schedule)",
            "WARNING: database connection pool saturation >90%",
        ],
        "desc": (
            "Task 5 — K8s Node OOM + Eviction Storm [Expert]: A Kubernetes worker node ran "
            "out of memory, triggering a pod eviction storm that took order-service offline. "
            "Query cluster metrics, inspect pod eviction logs, run DB connection diagnostics, "
            "rollback the order-service deployment that caused the memory regression, and "
            "resolve the incident."
        ),
        "services": {
            "cache": "healthy",
            "payment-gateway": "healthy",
            "cart-service": "degraded",
            "frontend": "degraded",
            "database": "critical",
            "api-gateway": "healthy",
            "order-service": "critical",
            "auth-service": "healthy",
        },
    },
}


class CloudSimulator:
    """
    Simulates a microservices cluster for SRE incident response training.

    Five incident scenarios with increasing complexity, each implementing
    a dense rubric grader that awards partial credit at each correct step.
    Wrong-service actions return realistic but unhelpful data (informative-
    negative design: agent sees real-looking data that reveals no root cause).

    Score boundaries: strictly in (BASE_SCORE, MAX_SCORE) — never 0 or 1.
    """

    BASE_SCORE = 0.001   # Minimum — never exactly 0.0
    MAX_SCORE  = 0.981   # Maximum — never exactly 1.0

    def __init__(self, task_level: int = 1):
        self.task_level = max(1, min(5, task_level))
        self.step_count = 0
        self.created_at = time.time()
        self.resolved = False
        self.output = "Environment initialized."
        self.score = self.BASE_SCORE
        self._rng = random.Random(42 + self.task_level)  # reproducible noise

        # Granular grader state — tracks which rubric items have been satisfied
        self.grader_state: Dict[str, bool] = {
            # Task 1
            "t1_queried_cache_memory": False,
            "t1_fetched_cache_logs": False,
            "t1_resolved": False,
            # Task 2
            "t2_listed_deployments": False,
            "t2_rolled_back_correctly": False,
            "t2_resolved": False,
            # Task 3
            "t3_traced_frontend": False,
            "t3_traced_cart": False,
            "t3_queried_db_activity": False,
            "t3_killed_deadlock": False,
            "t3_resolved": False,
            # Task 4
            "t4_queried_auth_metrics": False,
            "t4_fetched_auth_logs": False,
            "t4_rolled_back_auth": False,
            "t4_resolved": False,
            # Task 5
            "t5_queried_cluster_metrics": False,
            "t5_fetched_eviction_logs": False,
            "t5_ran_db_conn_query": False,
            "t5_rolled_back_order": False,
            "t5_resolved": False,
        }

        scenario = SCENARIOS[self.task_level]
        self.alerts:    List[str]       = scenario["alerts"]
        self.task_desc: str             = scenario["desc"]
        self._services_status: Dict[str, str] = scenario["services"].copy()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def services_status(self) -> Dict[str, str]:
        return self._services_status.copy()

    @property
    def live_metrics(self) -> Dict[str, float]:
        """Real-time telemetry with time-varying noise."""
        t = time.time()
        n = math.sin(t * 0.5) * 3 + math.cos(t * 1.3) * 1.5
        elapsed = min(180, t - self.created_at)

        cpu = round(max(0, 28 + n), 1)
        mem = round(max(0, 38 + n * 0.5), 1)
        err = round(max(0.1, 0.8 + n * 0.1), 1)
        lat = round(max(5, 45 + n * 2), 1)

        if not self.resolved:
            if self.task_level == 1:
                mem = round(min(99.0, 68 + elapsed * 0.18 + n), 1)
                cpu = round(min(75.0, 45 + elapsed * 0.10 + n), 1)
            elif self.task_level == 2:
                err = round(min(78.0, 22 + elapsed * 0.45 + n), 1)
                cpu = round(min(80.0, 50 + elapsed * 0.12 + n), 1)
            elif self.task_level == 3:
                cpu = round(min(99.0, 80 + elapsed * 0.12 + n), 1)
                err = round(min(55.0, 12 + elapsed * 0.28 + n), 1)
                mem = round(min(85.0, 55 + elapsed * 0.10 + n), 1)
                lat = round(min(35000, 5000 + elapsed * 180 + n * 100), 1)
            elif self.task_level == 4:
                cpu = round(min(99.0, 88 + elapsed * 0.06 + n), 1)
                err = round(min(65.0, 30 + elapsed * 0.20 + n), 1)
                lat = round(min(8000, 800 + elapsed * 40 + n * 50), 1)
            elif self.task_level == 5:
                mem  = round(min(99.0, 92 + elapsed * 0.04 + n), 1)
                cpu  = round(min(99.0, 85 + elapsed * 0.07 + n), 1)
                err  = round(min(90.0, 40 + elapsed * 0.30 + n), 1)
                lat  = round(min(15000, 2000 + elapsed * 70 + n * 80), 1)

        return {"cpu": cpu, "memory": mem, "error_rate": err, "latency_p99_ms": lat}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_score(self, points: float) -> None:
        self.score = min(self.MAX_SCORE, self.score + points)

    def _ts(self) -> str:
        """Current timestamp string."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _metric_series(self, base: float, spike: float, resolved: bool, label: str) -> str:
        """Generate a 10-point time-series as JSON."""
        points = []
        for i in range(10):
            t_off = (10 - i) * 10
            n = math.sin(i * 0.7) * 3
            val = (base + n) if resolved else min(100, spike + i * 3 + n)
            points.append({"time": f"T-{t_off}s", label: f"{round(val, 1)}%"})
        return json.dumps(points, indent=2)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def query_metrics(self, service: Optional[str], metric: Optional[str]) -> str:
        self.step_count += 1
        svc = (service or "").lower()
        met = (metric or "").lower()

        # Task 1 — cache memory spike
        if self.task_level == 1 and "cache" in svc and "memory" in met:
            if not self.grader_state["t1_queried_cache_memory"]:
                self.grader_state["t1_queried_cache_memory"] = True
                self._add_score(0.28)
            return self._metric_series(38, 68, self.resolved, "memory_%")

        # Task 2 — payment-gateway error rate (minor credit)
        if self.task_level == 2 and "payment" in svc and "error" in met:
            if not self.grader_state.get("t2_queried_metrics"):
                self.grader_state["t2_queried_metrics"] = True
                self._add_score(0.08)
            return self._metric_series(0.5, 22, self.resolved, "error_rate_%")

        # Task 4 — auth-service CPU (critical first step)
        if self.task_level == 4 and "auth" in svc and "cpu" in met:
            if not self.grader_state["t4_queried_auth_metrics"]:
                self.grader_state["t4_queried_auth_metrics"] = True
                self._add_score(0.20)
            return self._metric_series(30, 78, self.resolved, "cpu_%")

        # Task 5 — cluster/node memory (first step)
        if self.task_level == 5 and ("cluster" in svc or "node" in svc or "worker" in svc) and "memory" in met:
            if not self.grader_state["t5_queried_cluster_metrics"]:
                self.grader_state["t5_queried_cluster_metrics"] = True
                self._add_score(0.16)
            ts = self._ts()
            return json.dumps({
                "node": "worker-node-3",
                "memory_used_gi": 31.8,
                "memory_limit_gi": 32.0,
                "pods_evicted": 8,
                "oom_events": 3,
                "status": "OOMKilled",
                "timestamp": ts,
            }, indent=2)

        # Informative-negative: correct tool, wrong service/metric
        ts = self._ts()
        return json.dumps([
            {"time": f"T-{(10-i)*10}s", "value": f"{round(28 + math.sin(i)*3, 1)}%"}
            for i in range(10)
        ], indent=2)

    def fetch_logs(self, service: Optional[str], lines: Optional[int]) -> str:
        self.step_count += 1
        svc   = (service or "").lower()
        lines = max(1, min(200, lines or 20))
        ts    = self._ts()

        # Task 1 — cache OOM
        if self.task_level == 1 and "cache" in svc:
            if not self.grader_state["t1_fetched_cache_logs"]:
                self.grader_state["t1_fetched_cache_logs"] = True
                self._add_score(0.30)
            return (
                f"[{ts}] INFO  cache: Starting scheduled cache compaction (gen=3)\n"
                f"[{ts}] INFO  cache: Compaction progress: 24% ...\n"
                f"[{ts}] ERROR cache: java.lang.OutOfMemoryError: Java heap space\n"
                f"[{ts}] ERROR cache:   at com.cache.Compactor.compact(Compactor.java:847)\n"
                f"[{ts}] FATAL cache: Compaction aborted. Memory pressure critical.\n"
                f"[{ts}] WARN  cache: Heap utilization 95%. GC overhead limit exceeded.\n"
            )

        # Task 3 — frontend upstream timeout
        if self.task_level == 3 and "frontend" in svc:
            if not self.grader_state["t3_traced_frontend"]:
                self.grader_state["t3_traced_frontend"] = True
                self._add_score(0.16)
            return (
                f"[{ts}] INFO  frontend: GET /cart rendered in 78ms\n"
                f"[{ts}] WARN  frontend: Upstream cart-service response time >5s\n"
                f"[{ts}] ERROR frontend: Timeout upstream waiting for cart-service (30s exceeded)\n"
                f"[{ts}] ERROR frontend: HTTP 504 Gateway Timeout returned to client\n"
                f"[{ts}] ERROR frontend: Upstream cart-service unresponsive. Circuit breaker OPEN.\n"
            )

        # Task 3 — cart-service DB deadlock
        if self.task_level == 3 and ("cart" in svc):
            if not self.grader_state["t3_traced_cart"]:
                self.grader_state["t3_traced_cart"] = True
                self._add_score(0.16)
            return (
                f"[{ts}] INFO  cart-service: Processing GET /api/cart/user/4821\n"
                f"[{ts}] WARN  cart-service: DB connection pool exhausted (pool=20, waiting=17)\n"
                f"[{ts}] ERROR cart-service: DB connection timeout after 30000ms\n"
                f"[{ts}] ERROR cart-service: Transaction stalled — lock wait timeout exceeded\n"
                f"[{ts}] FATAL cart-service: Deadlock detected. PID 9942 holding locks. Rolling back.\n"
            )

        # Task 4 — auth-service hot-path JWT verify
        if self.task_level == 4 and "auth" in svc:
            if not self.grader_state["t4_fetched_auth_logs"]:
                self.grader_state["t4_fetched_auth_logs"] = True
                self._add_score(0.22)
            return (
                f"[{ts}] INFO  auth-service: JWT verification request received\n"
                f"[{ts}] WARN  auth-service: RSA key lookup P99 latency 2800ms (SLA=200ms)\n"
                f"[{ts}] ERROR auth-service: CPU throttled — goroutine pool saturated (2048/2048)\n"
                f"[{ts}] ERROR auth-service: Key cache miss rate 99.7% — regression in v2.3.1\n"
                f"[{ts}] FATAL auth-service: Request queue depth 8192. Shedding load.\n"
                f"[{ts}] INFO  auth-service: Version v2.3.1 deployed 47 mins ago.\n"
            )

        # Task 5 — pod eviction storm logs
        if self.task_level == 5 and ("order" in svc or "pod" in svc or "evict" in svc or "node" in svc):
            if not self.grader_state["t5_fetched_eviction_logs"]:
                self.grader_state["t5_fetched_eviction_logs"] = True
                self._add_score(0.18)
            return (
                f"[{ts}] WARN  kubelet worker-node-3: Memory pressure detected — 31.8Gi / 32Gi\n"
                f"[{ts}] ERROR kubelet worker-node-3: OOMKiller invoked — killing order-service-7f4b9 (2.1Gi RSS)\n"
                f"[{ts}] ERROR kubelet worker-node-3: Evicting pod order-service-5c8d2 (2.0Gi RSS)\n"
                f"[{ts}] ERROR kubelet worker-node-3: Evicting pod order-service-9e1f7 (1.9Gi RSS)\n"
                f"[{ts}] FATAL kubelet worker-node-3: 8 pods evicted. Node cordoned.\n"
                f"[{ts}] ERROR kube-scheduler: 8 pending pods cannot schedule — insufficient memory\n"
                f"[{ts}] INFO  order-service: version v3.2.0 deployed 23 mins ago (memory limit 2Gi, was 512Mi)\n"
            )

        # Informative-negative response
        return (
            f"[{ts}] INFO  {svc}: All operations nominal.\n"
            f"[{ts}] INFO  {svc}: Health check passed. Response time 12ms.\n"
            f"Showing last {lines} lines — no anomalies detected.\n"
        )

    def list_deployments(self, service: Optional[str]) -> str:
        self.step_count += 1
        svc = (service or "").lower()
        ts  = self._ts()

        # Task 2 — payment-gateway bad deployment
        if self.task_level == 2 and "payment" in svc:
            if not self.grader_state["t2_listed_deployments"]:
                self.grader_state["t2_listed_deployments"] = True
                self._add_score(0.22)
            return json.dumps([
                {"version": "v1.0.4", "status": "[CURRENT]",  "deployed_at": "47 mins ago", "replicas": 3, "error_rate": "42%"},
                {"version": "v1.0.3", "status": "stable",     "deployed_at": "3 days ago",  "replicas": 0, "error_rate": "0.2%"},
                {"version": "v1.0.2", "status": "deprecated",  "deployed_at": "2 weeks ago", "replicas": 0, "error_rate": "0.1%"},
            ], indent=2)

        # Task 4 — auth-service deployments
        if self.task_level == 4 and "auth" in svc:
            return json.dumps([
                {"version": "v2.3.1", "status": "[CURRENT]",  "deployed_at": "47 mins ago", "replicas": 6, "cpu_avg": "97%"},
                {"version": "v2.3.0", "status": "stable",     "deployed_at": "5 days ago",  "replicas": 0, "cpu_avg": "28%"},
                {"version": "v2.2.9", "status": "deprecated",  "deployed_at": "3 weeks ago", "replicas": 0, "cpu_avg": "25%"},
            ], indent=2)

        # Task 5 — order-service deployments
        if self.task_level == 5 and "order" in svc:
            return json.dumps([
                {"version": "v3.2.0", "status": "[CURRENT]",  "deployed_at": "23 mins ago", "replicas": 0, "memory_limit": "2Gi",  "note": "memory_limit increased from 512Mi — caused OOM on node"},
                {"version": "v3.1.9", "status": "stable",     "deployed_at": "2 days ago",  "replicas": 0, "memory_limit": "512Mi"},
                {"version": "v3.1.8", "status": "deprecated",  "deployed_at": "1 week ago",  "replicas": 0, "memory_limit": "512Mi"},
            ], indent=2)

        # Generic / wrong service
        return json.dumps([
            {"version": "v1.0.0", "status": "[CURRENT]", "deployed_at": "2 days ago", "replicas": 2},
        ], indent=2)

    def rollback_deployment(self, service: Optional[str], version: Optional[str]) -> str:
        self.step_count += 1
        svc = (service or "").lower()
        ver = (version or "").lower()
        ts  = self._ts()

        # Task 2 — correct rollback: payment-gateway → v1.0.3
        if self.task_level == 2 and "payment" in svc and "1.0.3" in ver:
            if not self.grader_state["t2_rolled_back_correctly"]:
                self.grader_state["t2_rolled_back_correctly"] = True
                self._add_score(0.35)
            self._services_status["payment-gateway"] = "healthy"
            self._services_status["frontend"] = "healthy"
            self._services_status["api-gateway"] = "healthy"
            return (
                f"[{ts}] Rollback initiated: payment-gateway v1.0.4 → v1.0.3\n"
                f"[{ts}] Draining traffic from v1.0.4 pods...\n"
                f"[{ts}] Starting v1.0.3 replicas (3/3 healthy)...\n"
                f"[{ts}] Rollback complete. v1.0.3 is now live.\n"
                f"[{ts}] Error rate: 42% → 0.3% ✓  (SLA restored)\n"
            )

        # Task 4 — correct rollback: auth-service → v2.3.0
        if self.task_level == 4 and "auth" in svc and "2.3.0" in ver:
            if not self.grader_state["t4_rolled_back_auth"]:
                self.grader_state["t4_rolled_back_auth"] = True
                self._add_score(0.28)
            self._services_status["api-gateway"] = "healthy"
            self._services_status["auth-service"] = "healthy"
            self._services_status["frontend"] = "healthy"
            self._services_status["payment-gateway"] = "healthy"
            self._services_status["cart-service"] = "healthy"
            return (
                f"[{ts}] Rollback initiated: auth-service v2.3.1 → v2.3.0\n"
                f"[{ts}] RSA key cache warming up...\n"
                f"[{ts}] Rollback complete. v2.3.0 is now live.\n"
                f"[{ts}] CPU: 97% → 27% ✓  JWT P99 latency: 2800ms → 180ms ✓\n"
            )

        # Task 5 — correct rollback: order-service → v3.1.9
        if self.task_level == 5 and "order" in svc and ("3.1.9" in ver or "v3.1" in ver):
            if not self.grader_state["t5_rolled_back_order"]:
                self.grader_state["t5_rolled_back_order"] = True
                self._add_score(0.24)
            self._services_status["order-service"] = "healthy"
            self._services_status["database"] = "healthy"
            self._services_status["cart-service"] = "healthy"
            self._services_status["frontend"] = "healthy"
            return (
                f"[{ts}] Rollback initiated: order-service v3.2.0 → v3.1.9\n"
                f"[{ts}] Memory limit reset to 512Mi per pod.\n"
                f"[{ts}] Uncordoning worker-node-3...\n"
                f"[{ts}] Scheduling 8 replacement pods...\n"
                f"[{ts}] Rollback complete. All order-service pods (3/3) healthy.\n"
                f"[{ts}] worker-node-3 memory: 31.8Gi → 18.2Gi ✓\n"
            )

        # Wrong rollback
        return f"[{ts}] ERROR: Deployment rollback failed — version '{version}' not found for service '{service}'.\n"

    def run_db_query(self, query: Optional[str]) -> str:
        self.step_count += 1
        q  = (query or "").lower()
        ts = self._ts()

        # Task 3 — query pg_stat_activity
        if self.task_level == 3 and "pg_stat_activity" in q:
            if not self.grader_state["t3_queried_db_activity"]:
                self.grader_state["t3_queried_db_activity"] = True
                self._add_score(0.18)
            return json.dumps([
                {"pid": 9942, "state": "active",  "wait_event_type": "Lock", "wait_event": "relation",
                 "query": "UPDATE cart_items SET ...", "duration_s": 312, "application_name": "cart-service"},
                {"pid": 9901, "state": "idle",    "wait_event_type": None,   "wait_event": None,
                 "query": "SELECT ...",             "duration_s": 2,   "application_name": "payment-gateway"},
                {"pid": 9880, "state": "active",  "wait_event_type": "Lock", "wait_event": "relation",
                 "query": "SELECT * FROM cart_items WHERE ...", "duration_s": 309, "application_name": "cart-service"},
            ], indent=2)

        # Task 3 — kill deadlock PID
        if self.task_level == 3 and "pg_terminate_backend" in q and "9942" in q:
            if not self.grader_state["t3_killed_deadlock"]:
                self.grader_state["t3_killed_deadlock"] = True
                self._add_score(0.22)
            self._services_status["database"] = "healthy"
            self._services_status["cart-service"] = "healthy"
            return json.dumps({
                "pg_terminate_backend": True,
                "pid_terminated": 9942,
                "message": "Lock released. Blocked transactions resumed.",
                "cart_service_recovery": "Connection pool recovering (17 → 2 waiting)",
            }, indent=2)

        # Task 5 — DB connection pool query
        if self.task_level == 5 and ("pg_stat_activity" in q or "connection" in q):
            if not self.grader_state["t5_ran_db_conn_query"]:
                self.grader_state["t5_ran_db_conn_query"] = True
                self._add_score(0.14)
            return json.dumps({
                "total_connections": 498,
                "max_connections": 500,
                "active": 312,
                "idle_in_transaction": 172,
                "waiting": 14,
                "note": "Connection pool near saturation — caused by order-service v3.2.0 connection leak",
            }, indent=2)

        return json.dumps({"result": [], "rows_affected": 0,
                           "message": "Query executed but returned no relevant results."}, indent=2)

    def resolve_incident(self, resolution_notes: Optional[str]) -> str:
        self.step_count += 1
        notes = (resolution_notes or "").lower()
        ts    = self._ts()

        resolved = False
        feedback = ""

        if self.task_level == 1:
            # Need: queried cache memory + fetched cache logs
            has_diag = self.grader_state["t1_queried_cache_memory"] and self.grader_state["t1_fetched_cache_logs"]
            oom_keywords = {"oom", "outofmemory", "heap", "memory", "compaction", "gc"}
            has_notes    = bool(oom_keywords & set(notes.split()))
            if (has_diag or has_notes) and not self.grader_state["t1_resolved"]:
                self.grader_state["t1_resolved"] = True
                self._add_score(0.33)
                resolved = True
                feedback = "Cache OOM confirmed. JVM heap limit increased. Compaction rescheduled."

        elif self.task_level == 2:
            has_diag = self.grader_state["t2_listed_deployments"] and self.grader_state["t2_rolled_back_correctly"]
            roll_kw  = {"rollback", "roll", "v1.0.3", "revert", "deployment"}
            has_notes = bool(roll_kw & set(notes.split()))
            if (has_diag or has_notes) and not self.grader_state["t2_resolved"]:
                self.grader_state["t2_resolved"] = True
                self._add_score(0.30)
                resolved = True
                feedback = "payment-gateway v1.0.4 rolled back to v1.0.3. Error rate normalised."

        elif self.task_level == 3:
            has_diag = (self.grader_state["t3_traced_frontend"]
                       and self.grader_state["t3_traced_cart"]
                       and self.grader_state["t3_killed_deadlock"])
            dead_kw  = {"deadlock", "pid", "9942", "lock", "terminate", "killed", "transaction"}
            has_notes = bool(dead_kw & set(notes.split()))
            if (has_diag or has_notes) and not self.grader_state["t3_resolved"]:
                self.grader_state["t3_resolved"] = True
                self._add_score(0.20)
                resolved = True
                feedback = "PID 9942 terminated. DB deadlock cleared. cart-service recovered."

        elif self.task_level == 4:
            has_diag = (self.grader_state["t4_queried_auth_metrics"]
                       and self.grader_state["t4_fetched_auth_logs"]
                       and self.grader_state["t4_rolled_back_auth"])
            auth_kw  = {"rollback", "auth", "v2.3.0", "cpu", "jwt", "cache", "regression"}
            has_notes = bool(auth_kw & set(notes.split()))
            if (has_diag or has_notes) and not self.grader_state["t4_resolved"]:
                self.grader_state["t4_resolved"] = True
                self._add_score(0.20)
                resolved = True
                feedback = "auth-service v2.3.1 rolled back to v2.3.0. CPU normalised. API gateway recovering."

        elif self.task_level == 5:
            has_diag = (self.grader_state["t5_queried_cluster_metrics"]
                       and self.grader_state["t5_fetched_eviction_logs"]
                       and self.grader_state["t5_rolled_back_order"])
            node_kw  = {"oom", "eviction", "order", "rollback", "memory", "v3.1.9", "node", "pod"}
            has_notes = bool(node_kw & set(notes.split()))
            if (has_diag or has_notes) and not self.grader_state["t5_resolved"]:
                self.grader_state["t5_resolved"] = True
                self._add_score(0.18)
                resolved = True
                feedback = "order-service v3.2.0 rolled back. Node OOM resolved. Pods rescheduled."

        if resolved:
            self.resolved = True
            self.alerts   = []
            for svc in self._services_status:
                if self._services_status[svc] != "healthy":
                    self._services_status[svc] = "healthy"
            return (
                f"[{ts}] RESOLVED: Incident confirmed closed.\n"
                f"[{ts}] {feedback}\n"
                f"[{ts}] All services nominal. Alerts cleared.\n"
                f"[{ts}] Post-incident score: {self.score:.4f}\n"
            )

        # Premature resolve — partial credit already earned, no resolve bonus
        return (
            f"[{ts}] WARN: resolve_incident called but root cause not fully established.\n"
            f"[{ts}] Please continue investigation. Active alerts: {self.alerts}\n"
            f"[{ts}] Current diagnostic score: {self.score:.4f}\n"
        )

    # ------------------------------------------------------------------
    # Main dispatcher
    # ------------------------------------------------------------------

    def step(self, action) -> str:
        """Dispatch an action to the appropriate handler and return output."""
        atype = getattr(action, "action_type", None) or (
            action.get("action_type") if isinstance(action, dict) else "noop"
        )
        atype = (atype or "noop").lower()

        if atype == "query_metrics":
            return self.query_metrics(
                getattr(action, "service", None) or (action.get("service") if isinstance(action, dict) else None),
                getattr(action, "metric", None) or (action.get("metric") if isinstance(action, dict) else None),
            )
        if atype == "fetch_logs":
            return self.fetch_logs(
                getattr(action, "service", None) or (action.get("service") if isinstance(action, dict) else None),
                getattr(action, "lines", None) or (action.get("lines") if isinstance(action, dict) else 20),
            )
        if atype == "list_deployments":
            return self.list_deployments(
                getattr(action, "service", None) or (action.get("service") if isinstance(action, dict) else None),
            )
        if atype == "rollback_deployment":
            return self.rollback_deployment(
                getattr(action, "service", None) or (action.get("service") if isinstance(action, dict) else None),
                getattr(action, "version", None) or (action.get("version") if isinstance(action, dict) else None),
            )
        if atype == "run_db_query":
            return self.run_db_query(
                getattr(action, "query", None) or (action.get("query") if isinstance(action, dict) else None),
            )
        if atype == "resolve_incident":
            return self.resolve_incident(
                getattr(action, "resolution_notes", None) or (action.get("resolution_notes") if isinstance(action, dict) else None),
            )
        # noop or unknown
        self.step_count += 1
        return f"[{self._ts()}] No-op. No action taken. (action_type={atype!r})"
