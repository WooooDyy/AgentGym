#!/usr/bin/env python3
"""
Comprehensive test for AgentEnv MCP with trajectory tracking and evaluation.

This script:
1. Starts the MCP server (or connects to running one)
2. Creates an environment
3. Runs an experiment with multiple episodes
4. Tracks trajectories (state, action, reward, done)
5. Collects evaluation metrics
"""

import requests
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import sys


@dataclass
class Transition:
    """Single transition in trajectory."""

    step: int
    action: str
    observation: str
    reward: float
    done: bool


@dataclass
class Episode:
    """Complete episode with trajectory."""

    episode_id: int
    data_idx: int
    transitions: List[Transition]
    total_reward: float
    total_steps: int
    success: bool


class TrajectoryCollector:
    """Collects and analyzes trajectories from environment."""

    def __init__(self):
        self.episodes: List[Episode] = []

    def add_episode(self, episode: Episode):
        """Add completed episode."""
        self.episodes.append(episode)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics across all episodes."""
        if not self.episodes:
            return {}

        total_rewards = [ep.total_reward for ep in self.episodes]
        total_steps = [ep.total_steps for ep in self.episodes]
        success_rate = sum(1 for ep in self.episodes if ep.success) / len(self.episodes)

        return {
            "num_episodes": len(self.episodes),
            "avg_reward": sum(total_rewards) / len(total_rewards),
            "min_reward": min(total_rewards),
            "max_reward": max(total_rewards),
            "avg_steps": sum(total_steps) / len(total_steps),
            "success_rate": success_rate,
            "total_transitions": sum(len(ep.transitions) for ep in self.episodes),
        }

    def save_to_json(self, filepath: str):
        """Save trajectories to JSON file."""
        data = {
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "data_idx": ep.data_idx,
                    "total_reward": ep.total_reward,
                    "total_steps": ep.total_steps,
                    "success": ep.success,
                    "transitions": [asdict(t) for t in ep.transitions],
                }
                for ep in self.episodes
            ],
            "statistics": self.get_statistics(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved trajectory data to {filepath}")


def run_episode(
    base_url: str, env_id: int, data_idx: int, actions: List[str], episode_id: int
) -> Episode:
    """
    Run a single episode and collect trajectory.

    Args:
        base_url: Base URL of environment server
        env_id: Environment ID
        data_idx: Data/task index
        actions: Sequence of actions to execute
        episode_id: Episode identifier

    Returns:
        Episode with complete trajectory
    """
    print(f"\n{'=' * 60}")
    print(f"Episode {episode_id}: Running with {len(actions)} actions")
    print(f"{'=' * 60}")

    # Reset environment
    resp = requests.post(f"{base_url}/reset", json={"id": env_id, "data_idx": data_idx})
    result = resp.json()

    if "error" in result:
        print(f"Error resetting: {result['error']}")
        return None

    print(f"Initial observation:\n{result['observation']}\n")

    transitions = []
    total_reward = 0.0

    # Execute actions
    for step, action in enumerate(actions, 1):
        print(f"Step {step}: Executing '{action}'")

        resp = requests.post(f"{base_url}/step", json={"id": env_id, "action": action})
        result = resp.json()

        if "error" in result:
            print(f"  Error: {result['error']}")
            break

        # Record transition
        transition = Transition(
            step=step,
            action=action,
            observation=result["observation"],
            reward=result["reward"],
            done=result["done"],
        )
        transitions.append(transition)

        # Print step info
        print(f"  Observation: {result['observation']}")
        print(f"  Reward: {result['reward']:.2f}")
        print(f"  Total Score: {result['score']:.2f}")
        print(f"  Done: {result['done']}")

        total_reward += result["reward"]

        if result["done"]:
            print(f"\nEpisode finished at step {step}")
            break

    # Create episode record
    episode = Episode(
        episode_id=episode_id,
        data_idx=data_idx,
        transitions=transitions,
        total_reward=total_reward,
        total_steps=len(transitions),
        success=total_reward > 0,  # Simple success criterion
    )

    print("\nEpisode Summary:")
    print(f"  Total Steps: {episode.total_steps}")
    print(f"  Total Reward: {episode.total_reward:.2f}")
    print(f"  Success: {episode.success}")

    return episode


def run_experiment(
    base_url: str, num_episodes: int = 3, actions_per_episode: List[List[str]] = None
):
    """
    Run complete experiment with multiple episodes.

    Args:
        base_url: Base URL of environment server
        num_episodes: Number of episodes to run
        actions_per_episode: List of action sequences for each episode
    """
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "AgentEnv MCP - Trajectory Experiment" + " " * 11 + "║")
    print("╚" + "═" * 58 + "╝")

    # Check server health
    print("\n1. Checking server health...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        health = resp.json()
        print(f"   ✓ Server healthy: {health}")
    except Exception as e:
        print(f"   ✗ Server not responding: {e}")
        print("   Make sure to start the server first:")
        print(
            "   uv run python -m agentenv_mcp.launch --example directional --port 8004"
        )
        return

    # Default action sequences if not provided
    if actions_per_episode is None:
        actions_per_episode = [
            ["right", "right", "up", "up"],  # Go to (2, 2)
            ["left", "left", "down"],  # Go to (-2, -1)
            ["up", "right", "up", "left"],  # Complex path
        ]

    # Create environment
    print("\n2. Creating environment...")
    resp = requests.post(f"{base_url}/create")
    env_id = resp.json()["id"]
    print(f"   ✓ Created environment ID: {env_id}")

    # Initialize trajectory collector
    collector = TrajectoryCollector()

    # Run episodes
    print(f"\n3. Running {num_episodes} episodes...")

    for episode_id in range(num_episodes):
        actions = actions_per_episode[episode_id % len(actions_per_episode)]
        episode = run_episode(
            base_url=base_url,
            env_id=env_id,
            data_idx=episode_id,
            actions=actions,
            episode_id=episode_id,
        )

        if episode:
            collector.add_episode(episode)

        # Small delay between episodes
        time.sleep(0.5)

    # Print statistics
    print(f"\n{'=' * 60}")
    print("Experiment Results")
    print(f"{'=' * 60}")

    stats = collector.get_statistics()
    print("\nStatistics:")
    print(f"  Total Episodes: {stats['num_episodes']}")
    print(f"  Total Transitions: {stats['total_transitions']}")
    print(f"  Average Reward: {stats['avg_reward']:.2f}")
    print(f"  Min Reward: {stats['min_reward']:.2f}")
    print(f"  Max Reward: {stats['max_reward']:.2f}")
    print(f"  Average Steps: {stats['avg_steps']:.1f}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")

    # Save trajectories
    trajectory_file = "trajectories.json"
    collector.save_to_json(trajectory_file)

    # Print sample trajectory
    print(f"\n{'=' * 60}")
    print("Sample Trajectory (Episode 0)")
    print(f"{'=' * 60}")

    if collector.episodes:
        episode = collector.episodes[0]
        print(f"\nEpisode ID: {episode.episode_id}")
        print(f"Total Reward: {episode.total_reward:.2f}")
        print(f"Total Steps: {episode.total_steps}\n")

        print("Transitions:")
        for trans in episode.transitions:
            print(f"  Step {trans.step}: {trans.action}")
            print(f"    → {trans.observation}")
            print(f"    Reward: {trans.reward:.2f}, Done: {trans.done}")

    # Close environment
    print("\n4. Cleaning up...")
    resp = requests.post(f"{base_url}/close", json={"id": env_id})
    print(f"   ✓ Closed environment {env_id}")

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "Experiment Completed!" + " " * 21 + "║")
    print("╚" + "═" * 58 + "╝\n")

    return collector


def main():
    """Main entry point."""
    # Configuration
    base_url = "http://localhost:8004"

    # Wait for server
    print("Waiting for server to be ready...")
    for i in range(10):
        try:
            requests.get(f"{base_url}/health", timeout=1)
            print("✓ Server is ready!\n")
            break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("⚠ Server not responding. Please start it first:")
        print(
            "  uv run python -m agentenv_mcp.launch --example directional --port 8004"
        )
        sys.exit(1)

    # Run experiment
    run_experiment(base_url=base_url, num_episodes=3)

    # Print final message
    print("\n📊 Trajectory data saved to trajectories.json")
    print("📈 View the data to analyze agent behavior and rewards")


if __name__ == "__main__":
    main()
