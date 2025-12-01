"""
Directional Navigation MCP Server using FastMCP.

A simple MCP server with 4 directional tools: up, down, left, right.
The agent navigates a 2D grid.

Usage:
    uv run python -m examples.directional.server
"""

from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("Directional Navigation")

# Global state for demonstration (in production, use proper state management)
_state = {"position": {"x": 0, "y": 0}, "history": [], "step_count": 0}


def _reset_state():
    """Reset state to initial values."""
    _state["position"] = {"x": 0, "y": 0}
    _state["history"] = []
    _state["step_count"] = 0


def _get_observation() -> str:
    """Get current observation."""
    return f"Current position: ({_state['position']['x']}, {_state['position']['y']}). Steps taken: {_state['step_count']}"


@mcp.tool()
def up() -> str:
    """Move up (increase y coordinate by 1)."""
    _state["position"]["y"] += 1
    _state["history"].append("up")
    _state["step_count"] += 1
    return f"Moved up. {_get_observation()}"


@mcp.tool()
def down() -> str:
    """Move down (decrease y coordinate by 1)."""
    _state["position"]["y"] -= 1
    _state["history"].append("down")
    _state["step_count"] += 1
    return f"Moved down. {_get_observation()}"


@mcp.tool()
def left() -> str:
    """Move left (decrease x coordinate by 1)."""
    _state["position"]["x"] -= 1
    _state["history"].append("left")
    _state["step_count"] += 1
    return f"Moved left. {_get_observation()}"


@mcp.tool()
def right() -> str:
    """Move right (increase x coordinate by 1)."""
    _state["position"]["x"] += 1
    _state["history"].append("right")
    _state["step_count"] += 1
    return f"Moved right. {_get_observation()}"


@mcp.tool()
def get_position() -> str:
    """Get current position without moving."""
    return _get_observation()


@mcp.tool()
def reset() -> str:
    """Reset the environment to initial state."""
    _reset_state()
    return f"Environment reset. {_get_observation()}"


if __name__ == "__main__":
    import os

    # Get port from environment variable or use default
    port = int(os.environ.get("MCP_PORT", 8001))
    host = os.environ.get("MCP_HOST", "0.0.0.0")

    print(f"Starting Directional Navigation MCP Server on {host}:{port}")
    print(f"SSE endpoint: http://{host}:{port}/sse")

    # Run with SSE transport
    mcp.run(transport="sse", host=host, port=port)
