#!/usr/bin/env python3
"""
End-to-end test for agentenv-mcp.

Demonstrates the complete workflow:
1. Start MCP server (directional navigation)
2. Connect agent via FastMCP client
3. Discover tools and execute actions
4. Record trajectory and report results
"""

import asyncio
import subprocess
import sys
import time

# Add parent to path for imports
sys.path.insert(0, ".")

from agentenv_mcp import MCPAgent, get_logger, setup_logging

logger = get_logger("test_e2e")


async def run_e2e_test(server_url: str, num_steps: int = 10) -> dict:
    """
    Run end-to-end test with the agent.

    Args:
        server_url: URL of MCP server
        num_steps: Number of steps to run

    Returns:
        Test results
    """
    logger.info(f"Starting E2E test with {num_steps} steps")

    agent = MCPAgent(server_url, max_steps=num_steps)

    # Discover available tools
    tools = await agent.discover_tools()
    logger.info(f"Discovered tools: {tools}")

    if not tools:
        return {"success": False, "error": "No tools discovered"}

    # Reset environment
    reset_obs = await agent.reset()
    logger.info(f"Reset: {reset_obs}")

    # Run episode
    trajectory = await agent.run_episode(policy="random")

    results = {
        "success": True,
        "tools": tools,
        "trajectory": trajectory.to_dict(),
        "steps_taken": len(trajectory.actions),
        "total_reward": trajectory.total_reward(),
    }

    logger.info(
        f"E2E test complete: {results['steps_taken']} steps, reward={results['total_reward']:.2f}"
    )
    return results


def start_server(port: int = 8001) -> subprocess.Popen:
    """Start the MCP server in background."""
    logger.info(f"Starting MCP server on port {port}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "agentenv_mcp.mcp_servers.directional"],
        env={**subprocess.os.environ, "MCP_PORT": str(port)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)  # Wait for server to start
    return proc


def main():
    """Run the E2E test."""
    setup_logging()
    logger.info("=" * 50)
    logger.info("AgentEnv MCP End-to-End Test")
    logger.info("=" * 50)

    port = 8001
    server_url = f"http://localhost:{port}/sse"

    # Start server
    server_proc = start_server(port)

    try:
        # Run test
        results = asyncio.run(run_e2e_test(server_url, num_steps=10))

        if results["success"]:
            print("\n" + "=" * 50)
            print("E2E TEST PASSED")
            print("=" * 50)
            print(f"Tools: {results['tools']}")
            print(f"Steps: {results['steps_taken']}")
            print(f"Total Reward: {results['total_reward']:.2f}")
            print("\nTrajectory:")
            for i, (action, obs) in enumerate(
                zip(
                    results["trajectory"]["actions"],
                    results["trajectory"]["observations"],
                )
            ):
                print(f"  {i + 1}. {action}: {obs[:60]}...")
        else:
            print(f"\nE2E TEST FAILED: {results.get('error', 'Unknown error')}")
            sys.exit(1)

    finally:
        # Cleanup
        logger.info("Stopping server...")
        server_proc.terminate()
        server_proc.wait(timeout=5)

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
