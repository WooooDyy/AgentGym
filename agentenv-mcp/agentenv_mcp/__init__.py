"""
AgentEnv MCP - Generic Environment with MCP Client/Server Architecture.

Core components:
- MCP Client: Connects to external MCP servers via SSE
- MCP Server: Base classes for implementing custom MCP servers
- Generic Environment: Uses MCP clients to provide tool-based action space
- Example implementations (directional navigation)

Two usage modes:

1. Internal mode (backward compatible):
    from agentenv_mcp import server
    from examples.directional import DirectionalToolSet, DirectionalState
    server.set_factories(DirectionalToolSet, DirectionalState)

2. MCP client mode (new):
    from agentenv_mcp import server_v2
    server_v2.set_mcp_config("mcp.json", default_client="directional")
"""

__version__ = "0.2.0"

# Core generic components (original)
from .environment import MCPEnvironment, MCPEnvServer, server
from .mcp_base import MCPState, MCPToolSet
from .server import app

# New MCP client components
from .mcp_client import MCPClient, MCPClientManager
from .mcp_environment import MCPBasedEnvironment, MCPEnvServerV2, server_v2

# Launch utilities
from .launch import launch
from .launch_mcp import launch_mcp

__all__ = [
    # Original components
    "MCPEnvironment",
    "MCPEnvServer",
    "server",
    "MCPState",
    "MCPToolSet",
    "app",
    "launch",
    # New MCP client components
    "MCPClient",
    "MCPClientManager",
    "MCPBasedEnvironment",
    "MCPEnvServerV2",
    "server_v2",
    "launch_mcp",
]
