"""
Directional Navigation MCP Server - Example Implementation.

A simple MCP server with 4 directional tools: up, down, left, right.
The agent navigates a 2D grid.
"""

from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentenv_mcp.mcp_base import MCPState, MCPToolSet


class DirectionalState(MCPState):
    """State for directional navigation environment."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.position = {"x": 0, "y": 0}
        self.history = []
        self.step_count = 0

    def get_observation(self) -> str:
        """Get current observation."""
        return f"Current position: ({self.position['x']}, {self.position['y']}). Steps taken: {self.step_count}"


class DirectionalToolSet(MCPToolSet):
    """Tool set for directional navigation."""

    def get_tools(self) -> Dict[str, callable]:
        """Get all available tools."""
        return {
            "up": self.tool_up,
            "down": self.tool_down,
            "left": self.tool_left,
            "right": self.tool_right,
            "get_position": self.tool_get_position,
        }

    def tool_up(self, state: DirectionalState) -> str:
        """Move up (increase y coordinate by 1)."""
        state.position["y"] += 1
        state.history.append("up")
        state.step_count += 1
        return f"Moved up. {state.get_observation()}"

    def tool_down(self, state: DirectionalState) -> str:
        """Move down (decrease y coordinate by 1)."""
        state.position["y"] -= 1
        state.history.append("down")
        state.step_count += 1
        return f"Moved down. {state.get_observation()}"

    def tool_left(self, state: DirectionalState) -> str:
        """Move left (decrease x coordinate by 1)."""
        state.position["x"] -= 1
        state.history.append("left")
        state.step_count += 1
        return f"Moved left. {state.get_observation()}"

    def tool_right(self, state: DirectionalState) -> str:
        """Move right (increase x coordinate by 1)."""
        state.position["x"] += 1
        state.history.append("right")
        state.step_count += 1
        return f"Moved right. {state.get_observation()}"

    def tool_get_position(self, state: DirectionalState) -> str:
        """Get current position."""
        return state.get_observation()

    def execute_tool(
        self, tool_name: str, state: MCPState, params: Dict[str, Any] = None
    ) -> str:
        """
        Execute a tool on the given state.

        Args:
            tool_name: Name of tool to execute
            state: State to operate on
            params: Tool parameters (unused for directional tools)

        Returns:
            Observation string
        """
        tools = self.get_tools()

        if tool_name not in tools:
            available = ", ".join(tools.keys())
            return f"Error: Unknown tool '{tool_name}'. Available tools: {available}"

        try:
            return tools[tool_name](state)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
