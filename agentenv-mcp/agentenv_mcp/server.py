"""
FastAPI Server - AgentGym API endpoints for MCP environment.

Exposes standard AgentGym endpoints:
- /create: Create new environment instance
- /step: Execute action (tool call)
- /reset: Reset environment
- /observation: Get current observation
- /close: Close environment
"""

from fastapi import FastAPI
from .model import StepRequestBody, ResetRequestBody, CloseRequestBody
from .environment import server


app = FastAPI(
    title="AgentEnv MCP",
    description="Gymnasium environment with internal MCP tools",
    version="0.1.0"
)


@app.get("/")
def hello():
    """Root endpoint."""
    return "AgentEnv MCP - Directional Navigation Environment"


@app.post("/create")
async def create():
    """
    Create a new environment instance.

    Returns:
        {"id": environment_id}
    """
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    """
    Execute an action (tool call) in the environment.

    Args:
        body: {id: env_id, action: tool_name}

    Returns:
        {observation, reward, score, done}
    """
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    """
    Reset environment to initial state.

    Args:
        body: {id: env_id, data_idx: task_index}

    Returns:
        {observation, reward, score, done}
    """
    return server.reset(body.id, body.data_idx)


@app.get("/observation")
def get_observation(id: int):
    """
    Get current observation without taking action.

    Args:
        id: Environment ID

    Returns:
        {observation, reward, score, done, deleted}
    """
    return server.observe(id)


@app.post("/close")
def close(body: CloseRequestBody):
    """
    Close and cleanup environment.

    Args:
        body: {id: env_id}

    Returns:
        bool (success status)
    """
    return server.close(body.id)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_envs": len(server.envs),
        "version": "0.1.0"
    }
