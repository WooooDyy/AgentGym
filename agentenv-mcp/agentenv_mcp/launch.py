"""
Launch script for AgentGym API Server.

Loads the MCP implementation and starts the server.
By default, uses the directional navigation example.
"""

import argparse
import uvicorn


def launch():
    """Entry point for launching the server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--example", type=str, default="directional",
                       help="Example to load (directional, ...)")
    args = parser.parse_args()

    # Load the MCP implementation
    print("=" * 60)
    print("AgentEnv MCP - Generic Environment Server")
    print("=" * 60)

    if args.example == "directional":
        from agentenv_mcp.environment import server
        from examples.directional import DirectionalToolSet, DirectionalState

        # Configure server with directional tools
        server.set_factories(DirectionalToolSet, DirectionalState)
        print(f"Loaded example: Directional Navigation")
        print(f"Tools: up, down, left, right, get_position")
    else:
        print(f"Unknown example: {args.example}")
        print("Available examples: directional")
        return

    print("=" * 60)
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    uvicorn.run("agentenv_mcp:app", host=args.host, port=args.port)


if __name__ == "__main__":
    launch()
