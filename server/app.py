import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
from server.models import CloudSREAction, CloudSREObservation
from server.simulator import CloudSimulator

app = FastAPI(title="CloudSRE OpenEnv")

current_sim: Optional[CloudSimulator] = None

class ResetRequest(BaseModel):
    task_id: int = 1

class StepResponse(BaseModel):
    observation: CloudSREObservation
    reward: float
    done: bool
    info: dict

class StateResponse(BaseModel):
    observation: CloudSREObservation
    score: float
    done: bool

@app.get("/")
async def root():
    return {"status": "ok", "message": "Cloud SRE OpenEnv API is running securely. Listening for agent requests on /reset and /step."}

@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()) -> StepResponse:
    global current_sim
    current_sim = CloudSimulator(task_level=req.task_id)
    obs = CloudSREObservation(
        active_alerts=current_sim.alerts,
        task_description=current_sim.task_desc,
        last_action_output="Environment Reset",
        echoed_message="Started new debug session."
    )
    return StepResponse(observation=obs, reward=0.0, done=False, info={})

@app.post("/step")
async def step(action: CloudSREAction) -> StepResponse:
    global current_sim
    if current_sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    output = current_sim.step(action)
    obs = CloudSREObservation(
        active_alerts=current_sim.alerts,
        task_description=current_sim.task_desc,
        last_action_output=output[:1000],
        echoed_message=f"Executed {action.action_type}"
    )
    return StepResponse(
        observation=obs, 
        reward=current_sim.score, 
        done=current_sim.resolved, 
        info={"steps": current_sim.step_count}
    )

@app.get("/state")
async def state() -> StateResponse:
    global current_sim
    if current_sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    obs = CloudSREObservation(
        active_alerts=current_sim.alerts,
        task_description=current_sim.task_desc,
        last_action_output=current_sim.output,
        echoed_message=""
    )
    return StateResponse(observation=obs, score=current_sim.score, done=current_sim.resolved)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
