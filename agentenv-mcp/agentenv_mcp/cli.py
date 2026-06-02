"""
CLI for agentenv-mcp.

Provides commands for running MCP servers and agents.
"""

import argparse
import asyncio
import sys

from .logging_config import get_logger, setup_logging

logger = get_logger("cli")


def run_server(args: argparse.Namespace) -> None:
    """Run an MCP server."""
    from .mcp_servers.directional import DirectionalMCPServer

    setup_logging()
    logger.info(f"Starting server on {args.host}:{args.port}")

    server = DirectionalMCPServer()
    server.run(transport="sse", host=args.host, port=args.port)


async def run_agent_async(args: argparse.Namespace) -> None:
    """Run the agent."""
    from .agent import run_agent_demo

    setup_logging()
    logger.info(f"Running agent against {args.server_url}")

    result = await run_agent_demo(args.server_url, num_steps=args.steps)

    print("\n=== Agent Demo Results ===")
    print(f"Server: {result['server_url']}")
    print(f"Tools discovered: {result['tools']}")
    print(f"Total reward: {result['trajectory']['total_reward']:.2f}")
    print(f"Steps taken: {len(result['trajectory']['actions'])}")
    print("\nTrajectory:")
    for i, (action, obs) in enumerate(
        zip(
            result["trajectory"]["actions"],
            result["trajectory"]["observations"],
        )
    ):
        print(f"  {i + 1}. {action}: {obs}")


def run_agent(args: argparse.Namespace) -> None:
    """Run the agent (sync wrapper)."""
    asyncio.run(run_agent_async(args))


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AgentEnv MCP - MCP integration for AgentGym"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run MCP server")
    server_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", type=int, default=8001, help="Port to bind to (default: 8001)"
    )
    server_parser.set_defaults(func=run_server)

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run agent demo")
    agent_parser.add_argument(
        "--server-url",
        default="http://localhost:8001/sse",
        help="MCP server URL (default: http://localhost:8001/sse)",
    )
    agent_parser.add_argument(
        "--steps", type=int, default=10, help="Number of steps (default: 10)"
    )
    agent_parser.set_defaults(func=run_agent)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
