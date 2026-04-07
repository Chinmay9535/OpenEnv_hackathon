from typing import Literal, Optional, List
from pydantic import BaseModel, Field

class CloudSREAction(BaseModel):
    action_type: Literal[
        "query_metrics", 
        "fetch_logs", 
        "list_deployments", 
        "rollback_deployment", 
        "run_db_query", 
        "resolve_incident",
        "noop"
    ] = Field(..., description="The type of action to perform.")
    
    service: Optional[str] = Field(None, description="Target microservice (e.g., 'cache', 'payment-gateway', 'cart-db', 'frontend').")
    metric: Optional[str] = Field(None, description="The metric to query (e.g., 'memory_usage', 'error_rate', 'cpu_usage').")
    lines: Optional[int] = Field(10, description="Number of recent log lines to fetch.")
    version: Optional[str] = Field(None, description="The deployment version to rollback to.")
    query: Optional[str] = Field(None, description="Database query to execute (e.g., 'SELECT ...', 'KILL ...').")
    resolution_notes: Optional[str] = Field(None, description="Notes describing the root cause and the fix applied.")

class CloudSREObservation(BaseModel):
    active_alerts: List[str] = Field(..., description="Currently active alerts firing in the cluster.")
    task_description: str = Field(..., description="The current objective.")
    last_action_output: str = Field(..., description="Output from the previous command.")
    echoed_message: str = Field("", description="System echoes.")
    live_metrics: dict = Field(default_factory=dict, description="Current real-time system metrics for the dashboard.")
