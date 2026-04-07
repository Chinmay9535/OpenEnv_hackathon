import json
import math
import time

class CloudSimulator:
    def __init__(self, task_level: int = 1):
        self.task_level = task_level
        self.step_count = 0
        self.created_at = time.time()
        self.resolved = False
        self.output = "Environment initialized. Awaiting commands."
        self.score = 0.0
        
        self.grader_state = {
            "queried_metrics": False,
            "fetched_logs": False,
            "checked_deployments": False,
            "killed_db_transaction": False,
            "rolled_back": False,
            "trace_frontend": False,
            "trace_cart": False
        }
        
        self.setup_scenario()
        
    def setup_scenario(self):
        if self.task_level == 1:
            self.alerts = ["CRITICAL: High Memory Usage on `cache` service"]
            self.task_desc = "Task 1 (Easy): The cache service has an alert. Check its logs and memory_usage metrics to diagnose, then submit a resolution."
        elif self.task_level == 2:
            self.alerts = ["CRITICAL: High 500 Error Rate on `payment-gateway`"]
            self.task_desc = "Task 2 (Medium): Payment gateway is failing. Check deployments and metrics, rollback if necessary, and submit a resolution."
        elif self.task_level == 3:
            self.alerts = ["CRITICAL: Frontend Timeout Spikes"]
            self.task_desc = "Task 3 (Hard): Users report timeouts. Trace the issue through logs and DB queries, resolve the root cause."
        else:
            self.alerts = []
            self.task_desc = "Unknown task level."

    @property
    def live_metrics(self):
        t = time.time()
        noise = math.sin(t * 0.5) * 5 + math.cos(t * 1.3) * 2
        
        # Base healthy state
        cpu = 30 + noise
        mem = 40 + noise * 0.5
        err = max(0, 1.0 + (noise * 0.2)) # 1% baseline error rate
        
        # Apply incident pressure if not resolved
        if not self.resolved:
            elapsed = min(120, t - self.created_at) # cap growth
            if self.task_level == 1:
                mem = min(99.9, 65 + (elapsed * 0.2) + noise)
            elif self.task_level == 2:
                err = min(80.0, 20 + (elapsed * 0.5) + noise)
            elif self.task_level == 3:
                cpu = min(100.0, 85 + (elapsed * 0.1) + noise)
                err = min(50.0, 10 + (elapsed * 0.3) + noise)
            
        return {
            "cpu": round(cpu, 1),
            "memory": round(mem, 1),
            "error_rate": round(err, 1)
        }

    def query_metrics(self, service: str, metric: str) -> str:
        self.step_count += 1
        
        points = []
        now = time.time()
        for i in range(10):
            t_offset = (10 - i) * 10 # 10s intervals
            t = now - t_offset
            noise = math.sin(t * 0.5) * 5
            
            val = 30 + noise # default
            if self.task_level == 1 and metric == 'memory_usage' and service == 'cache':
                self.grader_state['queried_metrics'] = True
                val = 40 + noise if (i < 5 and not self.resolved) else 90 + noise
                if self.resolved: val = 40 + noise
            elif self.task_level == 2 and metric == 'error_rate' and service == 'payment-gateway':
                self.grader_state['queried_metrics'] = True
                val = 1 + noise if (i < 5 and not self.resolved) else 50 + noise
                if self.resolved: val = 1 + noise
                
            val = min(max(val, 0), 100)
            points.append({"time": f"T-{t_offset}s", "val": f"{val:.1f}%"})
            
        return json.dumps(points)

    def fetch_logs(self, service: str, lines: int) -> str:
        self.step_count += 1
        curr_time = time.strftime("%H:%M:%S")
        if self.task_level == 1 and service == 'cache':
            self.grader_state['fetched_logs'] = True
            return f"{curr_time} INFO: Starting cache compaction\n{curr_time} ERROR: OutOfMemoryError in compaction routine."
        if self.task_level == 3:
            if service == 'frontend':
                self.grader_state['trace_frontend'] = True
                return f"{curr_time} ERROR: Timeout upstream waiting for cart-service."
            elif service == 'cart-service':
                self.grader_state['trace_cart'] = True
                return f"{curr_time} ERROR: DB connection timeout fetching cart items. Transaction stalled."
        return f"Fetching {lines} lines from {service} logs... All operations nominal."

    def list_deployments(self, service: str) -> str:
        self.step_count += 1
        if self.task_level == 2 and service == 'payment-gateway':
            self.grader_state['checked_deployments'] = True
            return "v1.0.4 - DEPLOYED RECENTLY\nv1.0.3 - STABLE deployed yesterday"
        return "v1.0.0 - STABLE"

    def rollback_deployment(self, service: str, version: str) -> str:
        self.step_count += 1
        if self.task_level == 2 and service == 'payment-gateway' and version == 'v1.0.3':
            self.alerts = []
            self.grader_state['rolled_back'] = True
            return f"Successfully rolled back {service} to {version}."
        return f"Failed to rollback {service} to {version}."

    def run_db_query(self, query: str) -> str:
        self.step_count += 1
        q = (query or "").lower()
        if self.task_level == 3:
            if "select" in q and "pg_stat_activity" in q:
                return "PID: 9942, state: active, query: 'UPDATE carts SET ...'"
            if "kill" in q or "pg_terminate_backend" in q:
                if "9942" in q:
                    self.alerts = []
                    self.grader_state['killed_db_transaction'] = True
                    return "Transaction 9942 killed. System recovered."
        return "Query executed with 0 rows affected."

    def resolve_incident(self, notes: str) -> str:
        self.step_count += 1
        self.resolved = True
        self.alerts = []
        self.score = 0.0
        
        # Grading logic
        if self.task_level == 1:
            if self.grader_state['queried_metrics']: self.score += 0.5
            if self.grader_state['fetched_logs']: self.score += 0.5
        elif self.task_level == 2:
            if self.grader_state['checked_deployments']: self.score += 0.3
            if self.grader_state['rolled_back']: self.score += 0.7
        elif self.task_level == 3:
            if self.grader_state['trace_frontend']: self.score += 0.2
            if self.grader_state['trace_cart']: self.score += 0.2
            if self.grader_state['killed_db_transaction']: self.score += 0.6
            
        return f"Incident resolved. Notes recorded: {notes}"

    def step(self, action):
        if self.resolved:
            return "Incident is already resolved."
            
        if action.action_type == "query_metrics":
            return self.query_metrics(action.service, action.metric)
        elif action.action_type == "fetch_logs":
            return self.fetch_logs(action.service, action.lines)
        elif action.action_type == "list_deployments":
            return self.list_deployments(action.service)
        elif action.action_type == "rollback_deployment":
            return self.rollback_deployment(action.service, action.version)
        elif action.action_type == "run_db_query":
            return self.run_db_query(action.query)
        elif action.action_type == "resolve_incident":
            return self.resolve_incident(action.resolution_notes)
        else:
            return "Unknown or NOOP action."
