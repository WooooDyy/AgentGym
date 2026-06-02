"""
Directional Navigation MCP Server.

A simple 2D grid navigation environment with up/down/left/right movements.
Demonstrates the MCP server pattern for RL environments.
"""

from dataclasses import dataclass, field

from ..logging_config import get_logger
from .base import BaseMCPServer, MCPServerState

logger = get_logger("mcp_servers.directional")


@dataclass
class DirectionalState(MCPServerState):
    """State for directional navigation environment."""

    x: int = 0
    y: int = 0
    history: list = field(default_factory=list)
    step_count: int = 0

    def reset(self) -> None:
        """Reset to origin."""
        self.x = 0
        self.y = 0
        self.history = []
        self.step_count = 0

    def get_observation(self) -> str:
        """Get current position observation."""
        return f"Position: ({self.x}, {self.y}), Steps: {self.step_count}"


class DirectionalMCPServer(BaseMCPServer):
    """
    MCP server for 2D directional navigation.

    Tools: up, down, left, right, get_position, reset
    """

    def __init__(self, name: str = "Directional Navigation"):
        super().__init__(name)

    def _create_default_state(self) -> DirectionalState:
        return DirectionalState()

    def _register_tools(self) -> None:
        """Register directional movement tools."""

        @self.mcp.tool()
        def up() -> str:
            """Move up (increase y by 1)."""
            self.state.y += 1
            self.state.step_count += 1
            self.state.history.append("up")
            obs = self.state.get_observation()
            logger.debug(f"up -> {obs}")
            return f"Moved up. {obs}"

        @self.mcp.tool()
        def down() -> str:
            """Move down (decrease y by 1)."""
            self.state.y -= 1
            self.state.step_count += 1
            self.state.history.append("down")
            obs = self.state.get_observation()
            logger.debug(f"down -> {obs}")
            return f"Moved down. {obs}"

        @self.mcp.tool()
        def left() -> str:
            """Move left (decrease x by 1)."""
            self.state.x -= 1
            self.state.step_count += 1
            self.state.history.append("left")
            obs = self.state.get_observation()
            logger.debug(f"left -> {obs}")
            return f"Moved left. {obs}"

        @self.mcp.tool()
        def right() -> str:
            """Move right (increase x by 1)."""
            self.state.x += 1
            self.state.step_count += 1
            self.state.history.append("right")
            obs = self.state.get_observation()
            logger.debug(f"right -> {obs}")
            return f"Moved right. {obs}"

        @self.mcp.tool()
        def get_position() -> str:
            """Get current position without moving."""
            return self.state.get_observation()

        @self.mcp.tool()
        def reset() -> str:
            """Reset environment to initial state."""
            self.state.reset()
            logger.info("Environment reset")
            return f"Reset complete. {self.state.get_observation()}"


def create_server(host: str = "0.0.0.0", port: int = 8001) -> DirectionalMCPServer:
    """
    Create and configure a directional MCP server.

    Args:
        host: Host to bind to
        port: Port to bind to

    Returns:
        Configured DirectionalMCPServer instance
    """
    server = DirectionalMCPServer()
    return server


if __name__ == "__main__":
    import os

    port = int(os.environ.get("MCP_PORT", 8001))
    host = os.environ.get("MCP_HOST", "0.0.0.0")

    server = create_server(host, port)
    server.run(transport="sse", host=host, port=port)
