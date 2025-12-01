#!/usr/bin/env python3
"""
Start script for launching multiple MCP servers.

Usage:
    # Start default server (directional)
    python examples/start_servers.py

    # Start specific server
    python examples/start_servers.py directional

    # Start multiple servers (planned for future)
    python examples/start_servers.py directional custom
"""

import argparse
import sys
import subprocess


AVAILABLE_SERVERS = {
    "directional": {
        "module": "examples.directional.server",
        "port": 8001,
        "description": "Directional navigation MCP server",
    }
}


def start_server(server_name: str, port: int = None):
    """
    Start an MCP server.

    Args:
        server_name: Name of the server to start
        port: Port to run on (optional, uses default if not specified)
    """
    if server_name not in AVAILABLE_SERVERS:
        print(f"Error: Unknown server '{server_name}'")
        print(f"Available servers: {', '.join(AVAILABLE_SERVERS.keys())}")
        return False

    server_info = AVAILABLE_SERVERS[server_name]
    module = server_info["module"]
    default_port = server_info["port"]

    if port is None:
        port = default_port

    print("=" * 60)
    print(f"Starting MCP Server: {server_name}")
    print(f"Description: {server_info['description']}")
    print(f"Module: {module}")
    print(f"Port: {port}")
    print("=" * 60)

    try:
        # Start the server
        cmd = ["uv", "run", "python", "-m", module]

        # Set environment variable for port if needed
        # (servers should read PORT environment variable)
        import os

        env = os.environ.copy()
        env["MCP_PORT"] = str(port)

        print(f"\nRunning: {' '.join(cmd)}")
        print(f"Server will be available at: http://localhost:{port}/sse")
        print("\nPress Ctrl+C to stop the server\n")

        process = subprocess.run(cmd, env=env)
        return process.returncode == 0

    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        return True
    except Exception as e:
        print(f"Error starting server: {e}")
        return False


def list_servers():
    """List all available MCP servers."""
    print("\nAvailable MCP Servers:")
    print("=" * 60)
    for name, info in AVAILABLE_SERVERS.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Module: {info['module']}")
        print(f"  Default Port: {info['port']}")
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start MCP servers for AgentGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "servers",
        nargs="*",
        default=["directional"],
        help="Server(s) to start (default: directional)",
    )

    parser.add_argument("--list", action="store_true", help="List available servers")

    parser.add_argument("--port", type=int, help="Port to run on (overrides default)")

    args = parser.parse_args()

    if args.list:
        list_servers()
        return 0

    if not args.servers:
        print("Error: No server specified")
        list_servers()
        return 1

    if len(args.servers) > 1:
        print("Note: Multi-server mode not yet fully implemented.")
        print("Starting first server only...\n")

    # Start the first server
    server_name = args.servers[0]
    success = start_server(server_name, args.port)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
