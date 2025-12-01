"""
Launch script for AgentGym API Server in MCP Client Mode.

Uses MCP clients to connect to external MCP servers via SSE.
"""

import argparse
import uvicorn
from pathlib import Path


def launch_mcp():
    """Entry point for launching the server in MCP client mode."""
    parser = argparse.ArgumentParser(
        description="AgentGym MCP Environment - Client Mode"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run AgentGym server on"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--config",
        type=str,
        default="agentenv_mcp/mcp.json",
        help="Path to mcp.json configuration file",
    )
    parser.add_argument(
        "--default-client", type=str, default=None, help="Default MCP client to use"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found")
        print("Using example config from mcp.example.json")
        config_path = Path("agentenv_mcp/mcp.example.json")

    print("=" * 60)
    print("AgentGym MCP Environment - Client Mode")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    # Initialize MCP clients
    from agentenv_mcp.mcp_environment import server_v2

    server_v2.set_mcp_config(str(config_path), args.default_client)

    print("MCP Clients initialized")
    print("=" * 60)

    # Start server
    from agentenv_mcp import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    launch_mcp()
