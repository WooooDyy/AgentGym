"""
AgentEnv MCP - Generic Environment with Pluggable MCP Tools.

Core components:
- Generic MCP environment (works with any tool set)
- Base classes for implementing custom MCP servers
- Example implementations (directional navigation)

To use a specific MCP implementation, configure the server:

    from agentenv_mcp import server
    from examples.directional import DirectionalToolSet, DirectionalState

    server.set_factories(DirectionalToolSet, DirectionalState)
"""

__version__ = "0.1.0"

# Core generic components
from .environment import MCPEnvironment, MCPEnvServer, server
from .mcp_base import MCPState, MCPToolSet
from .server import app

# Launch utility
from .launch import launch

__all__ = [
    "MCPEnvironment",
    "MCPEnvServer",
    "server",
    "MCPState",
    "MCPToolSet",
    "app",
    "launch",
]
