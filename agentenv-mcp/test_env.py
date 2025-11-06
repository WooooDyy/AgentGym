"""
Test script for AgentEnv MCP - Directional Navigation Environment.

Demonstrates:
1. Creating an environment
2. Taking actions (tool calls)
3. Observing state changes
4. Receiving rewards
"""

import requests
import time


def test_environment():
    """Test the environment via HTTP API."""
    base_url = "http://localhost:8004"

    print("\n" + "=" * 60)
    print("Testing AgentEnv MCP - Directional Navigation")
    print("=" * 60)

    # Check health
    print("\n1. Health check...")
    resp = requests.get(f"{base_url}/health")
    print(f"   Status: {resp.json()}")

    # Create environment
    print("\n2. Creating environment...")
    resp = requests.post(f"{base_url}/create")
    env_id = resp.json()["id"]
    print(f"   Created environment ID: {env_id}")

    # Reset environment
    print("\n3. Resetting environment...")
    resp = requests.post(f"{base_url}/reset", json={"id": env_id, "data_idx": 0})
    result = resp.json()
    print(f"   Initial observation:\n   {result['observation']}")

    # Define a sequence of actions to reach position (2, 2)
    actions = ["right", "right", "up", "up", "get_position"]

    print(f"\n4. Executing actions to reach (2, 2)...")
    total_reward = 0.0

    for i, action in enumerate(actions):
        print(f"\n   Step {i + 1}: {action}")
        print(f"   " + "-" * 56)

        resp = requests.post(f"{base_url}/step", json={"id": env_id, "action": action})
        result = resp.json()

        if "error" in result:
            print(f"   Error: {result['error']}")
            break

        print(f"   Observation: {result['observation']}")
        print(f"   Reward: {result['reward']:.2f}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Done: {result['done']}")

        total_reward += result['reward']

        if result['done']:
            print(f"\n   Episode finished!")
            break

    # Get observation
    print(f"\n5. Getting final observation...")
    resp = requests.get(f"{base_url}/observation", params={"id": env_id})
    result = resp.json()
    print(f"   {result['observation']}")
    print(f"   Total score: {result['score']:.2f}")

    # Close environment
    print(f"\n6. Closing environment...")
    resp = requests.post(f"{base_url}/close", json={"id": env_id})
    print(f"   Closed: {resp.json()}")

    print("\n" + "=" * 60)
    print(f"✓ Test completed! Total reward: {total_reward:.2f}")
    print("=" * 60)


def test_multiple_instances():
    """Test multiple environment instances."""
    base_url = "http://localhost:8004"

    print("\n" + "=" * 60)
    print("Testing Multiple Environment Instances")
    print("=" * 60)

    # Create two environments
    resp1 = requests.post(f"{base_url}/create")
    env1 = resp1.json()["id"]

    resp2 = requests.post(f"{base_url}/create")
    env2 = resp2.json()["id"]

    print(f"\nCreated environments: {env1}, {env2}")

    # Reset both
    requests.post(f"{base_url}/reset", json={"id": env1, "data_idx": 0})
    requests.post(f"{base_url}/reset", json={"id": env2, "data_idx": 0})

    # Move env1 to the right
    print(f"\nEnvironment {env1}: Moving right")
    resp = requests.post(f"{base_url}/step", json={"id": env1, "action": "right"})
    print(f"  {resp.json()['observation']}")

    # Move env2 upward
    print(f"\nEnvironment {env2}: Moving up")
    resp = requests.post(f"{base_url}/step", json={"id": env2, "action": "up"})
    print(f"  {resp.json()['observation']}")

    # Verify they maintain separate states
    resp1 = requests.get(f"{base_url}/observation", params={"id": env1})
    resp2 = requests.get(f"{base_url}/observation", params={"id": env2})

    print(f"\nFinal positions:")
    print(f"  Environment {env1}: {resp1.json()['observation']}")
    print(f"  Environment {env2}: {resp2.json()['observation']}")

    # Close both
    requests.post(f"{base_url}/close", json={"id": env1})
    requests.post(f"{base_url}/close", json={"id": env2})

    print("\n✓ Multiple instances test passed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "AgentEnv MCP Tests" + " " * 24 + "║")
    print("╚" + "═" * 58 + "╝")

    # Wait a bit for server to start if just launched
    print("\nWaiting for server to be ready...")
    for i in range(5):
        try:
            requests.get("http://localhost:8004/health", timeout=1)
            print("✓ Server is ready!\n")
            break
        except:
            time.sleep(1)
    else:
        print("⚠ Server not responding. Make sure to start it first:")
        print("  uv run python -m agentenv_mcp.launch --example directional --port 8004")
        exit(1)

    # Run tests
    try:
        test_environment()
        test_multiple_instances()

        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + " " * 18 + "All Tests Passed!" + " " * 22 + "║")
        print("╚" + "═" * 58 + "╝\n")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
